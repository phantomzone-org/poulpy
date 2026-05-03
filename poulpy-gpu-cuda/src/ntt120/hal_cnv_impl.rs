//! HalConvolutionImpl for `CudaNtt120Backend`.
//!
//! Memory layout conventions (matching `bytes_of_*` in `mod.rs`):
//!
//! - `CnvPVecL` = VecZnxDft layout: `[cols × size × 4 × n]` i32 (prime-major).
//! - `CnvPVecR` = SvpPPol layout:   `[cols × size × 4 × n × 2]` u32
//!   (prime-major, 2 Montgomery lanes per coeff).
//!
//! This lets `cnv_prepare_left` reuse `ntt120_ntt_fwd_apply` directly, and
//! `cnv_prepare_right` reuse `ntt120_ntt_fwd_apply` + `ntt120_svp_c_from_b`.

use std::mem::size_of;

use poulpy_hal::{
    api::{ScratchArenaTakeBasic, VecZnxExtractCoeffBackend},
    layouts::{
        Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, DataView, DataViewMut, Module,
        ScratchArena, VecZnx, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
    },
    oep::HalConvolutionImpl,
};

use super::hal_impl::{buf_device_ptr, slice_device_ptr};
use crate::ntt120::{CudaNtt120Backend, types::Big32};
use crate::{CudaBuf, cuda_stream};

// ── extern "C" kernel launchers ───────────────────────────────────────────────

unsafe extern "C" {
    fn ntt120_ntt_fwd_apply(
        stream: *mut std::ffi::c_void,
        dst: *mut i32,
        src: *const i64,
        twiddle_fwd: *const u32,
        twiddle_fwd_msb: *const u32,
        primes: *const u32,
        inv_primes: *const i32,
        log_n: i32,
        batch: i32,
    );

    fn ntt120_svp_c_from_b(stream: *mut std::ffi::c_void, dst: *mut u32, src: *const i32, n: i32, cols: i32, primes: *const u32);

    fn ntt120_apply_mask_i64(stream: *mut std::ffi::c_void, dst: *mut i64, src: *const i64, mask: i64, n: i32);

    fn ntt120_cnv_apply_dft(
        stream: *mut std::ffi::c_void,
        res: *mut i32,
        a: *const i32,
        b: *const u32,
        n: i32,
        a_size: i32,
        b_size: i32,
        active_ovs: i32,
        cnv_offset: i32,
        primes: *const u32,
    );

    fn ntt120_cnv_pairwise_apply_dft(
        stream: *mut std::ffi::c_void,
        res: *mut i32,
        a_i: *const i32,
        a_j: *const i32,
        b_i: *const u32,
        b_j: *const u32,
        n: i32,
        a_size: i32,
        b_size: i32,
        active_ovs: i32,
        cnv_offset: i32,
        primes: *const u32,
    );

    fn ntt120_cnv_by_const_apply(
        stream: *mut std::ffi::c_void,
        res: *mut u32,
        a: *const i64,
        b: *const i64,
        n: i32,
        a_size: i32,
        b_size: i32,
        active_ovs: i32,
        cnv_offset: i32,
    );

    fn cudaMemsetAsync(
        dev_ptr: *mut std::ffi::c_void,
        value: std::ffi::c_int,
        count: usize,
        stream: *mut std::ffi::c_void,
    ) -> std::ffi::c_int;
}

// ── Byte-size helpers ─────────────────────────────────────────────────────────

/// Bytes per limb in CnvPVecL = VecZnxDft limb = 4*n*sizeof(i32).
#[inline]
fn cnv_l_limb_bytes(n: usize) -> usize {
    4 * n * size_of::<i32>()
}

/// Bytes per limb in CnvPVecR = SvpPPol column = 4*n*2*sizeof(u32).
#[inline]
fn cnv_r_limb_bytes(n: usize) -> usize {
    4 * n * 2 * size_of::<u32>()
}

/// Bytes per limb in VecZnx = n*sizeof(i64).
#[inline]
fn vec_znx_limb_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

/// Bytes per limb in VecZnxBig = n*sizeof(Big32).
#[inline]
fn vec_znx_big_limb_bytes(n: usize) -> usize {
    n * size_of::<Big32>()
}

// ── Common NTT prepare helper ─────────────────────────────────────────────────
//
// Runs `ntt120_ntt_fwd_apply` on one column of a VecZnx into a NTT output
// buffer (prime-major [batch × 4 × n] i32).  Handles the optional masking of
// the last active limb before the NTT.
//
// Parameters:
//   a_col_ptr  — device pointer to column start of VecZnx (i64, limb-major)
//   dst_ptr    — device pointer to column start of NTT output (i32, prime-major)
//   n          — ring dimension
//   min_size   — number of active input limbs to transform
//   mask       — i64 AND mask applied to the last limb (-1 = no masking)
//   handle/stream pointers — standard NTT parameters

#[allow(clippy::too_many_arguments)]
unsafe fn launch_ntt_prepare(
    stream_raw: *mut std::ffi::c_void,
    dst_ptr: *mut i32,
    a_ptr: *const i64,
    tw_fwd: *const u32,
    tw_fwd_msb: *const u32,
    primes_ptr: *const u32,
    inv_primes_ptr: *const i32,
    log_n: i32,
    n: usize,
    min_size: usize,
    mask: i64,
) -> Option<CudaBuf> {
    // No work if there's nothing to transform.
    if min_size == 0 {
        return None;
    }

    let unmasked = if mask == -1 { min_size } else { min_size.saturating_sub(1) };

    // NTT unmasked limbs (may be all of them when mask == -1).
    if unmasked > 0 {
        unsafe {
            ntt120_ntt_fwd_apply(
                stream_raw,
                dst_ptr,
                a_ptr,
                tw_fwd,
                tw_fwd_msb,
                primes_ptr,
                inv_primes_ptr,
                log_n,
                unmasked as i32,
            );
        }
    }

    // Masked last limb.
    if mask != -1 {
        // Copy and mask the last i64 limb into a temp buffer.
        let tmp = CudaNtt120Backend::alloc_bytes(n * size_of::<i64>());
        let tmp_ptr = tmp.raw_device_ptr(0) as *mut i64;

        let last_a = unsafe { (a_ptr as *const u8).add(unmasked * n * size_of::<i64>()) as *const i64 };
        unsafe {
            ntt120_apply_mask_i64(stream_raw, tmp_ptr, last_a, mask, n as i32);
        }

        let last_dst = unsafe { dst_ptr.add(unmasked * 4 * n) };
        unsafe {
            ntt120_ntt_fwd_apply(
                stream_raw,
                last_dst,
                tmp_ptr as *const i64,
                tw_fwd,
                tw_fwd_msb,
                primes_ptr,
                inv_primes_ptr,
                log_n,
                1,
            );
        }
        // Return the temp buf so the caller can sync-then-drop.
        return Some(tmp);
    }

    None
}

// ── HalConvolutionImpl ────────────────────────────────────────────────────────

unsafe impl HalConvolutionImpl<CudaNtt120Backend> for CudaNtt120Backend {
    // ── scratch sizes ────────────────────────────────────────────────────────

    fn cnv_prepare_left_tmp_bytes(_module: &Module<CudaNtt120Backend>, _res_size: usize, _a_size: usize) -> usize {
        0
    }

    fn cnv_prepare_right_tmp_bytes(_module: &Module<CudaNtt120Backend>, _res_size: usize, _a_size: usize) -> usize {
        0
    }

    fn cnv_apply_dft_tmp_bytes(
        _module: &Module<CudaNtt120Backend>,
        _cnv_offset: usize,
        _res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    fn cnv_by_const_apply_tmp_bytes(
        _module: &Module<CudaNtt120Backend>,
        _cnv_offset: usize,
        _res_size: usize,
        _a_size: usize,
        b_size: usize,
    ) -> usize {
        VecZnx::<Vec<u8>>::bytes_of(1, 1, b_size)
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        _module: &Module<CudaNtt120Backend>,
        _cnv_offset: usize,
        _res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    fn cnv_prepare_self_tmp_bytes(_module: &Module<CudaNtt120Backend>, _res_size: usize, _a_size: usize) -> usize {
        0
    }

    // ── cnv_prepare_left ─────────────────────────────────────────────────────
    //
    // Encodes VecZnx (i64 coefficients) → CnvPVecL (prime-major i32 NTT output,
    // same format as VecZnxDft).  Mask is applied to the last active limb.

    fn cnv_prepare_left<'s, 'r>(
        module: &Module<CudaNtt120Backend>,
        res: &mut CnvPVecLBackendMut<'r, CudaNtt120Backend>,
        a: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        mask: i64,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        let log_n = handle.log_n as i32;

        let res_size = res.size();
        let min_size = res_size.min(a.size());

        let tw_fwd = slice_device_ptr(&handle.twiddle_fwd) as *const u32;
        let tw_fwd_msb = handle
            .twiddle_fwd_msb
            .as_ref()
            .map(|s| slice_device_ptr(s) as *const u32)
            .unwrap_or(std::ptr::null());
        let primes_ptr = slice_device_ptr(&handle.primes) as *const u32;
        let inv_primes_ptr = slice_device_ptr(&handle.inv_primes) as *const i32;

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let a_data = a.data;
        let a_buf: &CudaBuf = unsafe { a_data.ptr.as_ref() };

        let cols = res.cols();
        let res_data = res.data_mut();
        let res_buf: &CudaBuf = unsafe { res_data.ptr.as_ref() };

        let mut tmp_bufs: Vec<CudaBuf> = Vec::new();

        for col in 0..cols {
            let a_col_offset = a_data.offset + col * a.size() * vec_znx_limb_bytes(n);
            let res_col_offset = res_data.offset + col * res_size * cnv_l_limb_bytes(n);

            let a_ptr = buf_device_ptr(a_buf, a_col_offset) as *const i64;
            let res_ptr = buf_device_ptr(res_buf, res_col_offset) as *mut i32;

            let tmp = unsafe {
                launch_ntt_prepare(
                    stream_raw,
                    res_ptr,
                    a_ptr,
                    tw_fwd,
                    tw_fwd_msb,
                    primes_ptr,
                    inv_primes_ptr,
                    log_n,
                    n,
                    min_size,
                    mask,
                )
            };
            if let Some(b) = tmp {
                tmp_bufs.push(b);
            }

            // Zero extra limbs beyond min_size.
            let extra = res_size.saturating_sub(min_size);
            if extra > 0 {
                let zero_ptr = buf_device_ptr(res_buf, res_col_offset + min_size * cnv_l_limb_bytes(n)) as *mut std::ffi::c_void;
                unsafe {
                    cudaMemsetAsync(zero_ptr, 0, extra * cnv_l_limb_bytes(n), stream_raw);
                }
            }
        }

        if !tmp_bufs.is_empty() {
            stream.synchronize().expect("CUDA sync failed in cnv_prepare_left");
        }
        drop(tmp_bufs);
    }

    // ── cnv_prepare_right ────────────────────────────────────────────────────
    //
    // Encodes VecZnx → CnvPVecR (prime-major u32 with 2 Montgomery lanes).
    // NTT output is first generated into an intermediate device buffer, then
    // converted to the Montgomery pair format via `ntt120_svp_c_from_b`.

    fn cnv_prepare_right<'s, 'r>(
        module: &Module<CudaNtt120Backend>,
        res: &mut CnvPVecRBackendMut<'r, CudaNtt120Backend>,
        a: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        mask: i64,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        let log_n = handle.log_n as i32;

        let res_size = res.size();
        let min_size = res_size.min(a.size());

        let tw_fwd = slice_device_ptr(&handle.twiddle_fwd) as *const u32;
        let tw_fwd_msb = handle
            .twiddle_fwd_msb
            .as_ref()
            .map(|s| slice_device_ptr(s) as *const u32)
            .unwrap_or(std::ptr::null());
        let primes_ptr = slice_device_ptr(&handle.primes) as *const u32;
        let inv_primes_ptr = slice_device_ptr(&handle.inv_primes) as *const i32;

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let a_data = a.data;
        let a_buf: &CudaBuf = unsafe { a_data.ptr.as_ref() };

        let cols = res.cols();
        let res_data = res.data_mut();
        let res_buf: &CudaBuf = unsafe { res_data.ptr.as_ref() };

        // Temp buffer for NTT output (prime-major i32) before c_from_b conversion.
        let ntt_tmp = if min_size > 0 {
            Some(CudaNtt120Backend::alloc_bytes(min_size * cnv_l_limb_bytes(n)))
        } else {
            None
        };

        let mut mask_tmps: Vec<CudaBuf> = Vec::new();

        for col in 0..cols {
            let a_col_offset = a_data.offset + col * a.size() * vec_znx_limb_bytes(n);
            let res_col_offset = res_data.offset + col * res_size * cnv_r_limb_bytes(n);

            let a_ptr = buf_device_ptr(a_buf, a_col_offset) as *const i64;
            let res_ptr = buf_device_ptr(res_buf, res_col_offset) as *mut u32;

            if min_size > 0 {
                let ntt_buf = ntt_tmp.as_ref().unwrap();
                let ntt_ptr = ntt_buf.raw_device_ptr(0) as *mut i32;

                let tmp = unsafe {
                    launch_ntt_prepare(
                        stream_raw,
                        ntt_ptr,
                        a_ptr,
                        tw_fwd,
                        tw_fwd_msb,
                        primes_ptr,
                        inv_primes_ptr,
                        log_n,
                        n,
                        min_size,
                        mask,
                    )
                };
                if let Some(b) = tmp {
                    mask_tmps.push(b);
                }

                // Convert NTT output → CnvPVecR (Montgomery pairs).
                unsafe {
                    ntt120_svp_c_from_b(
                        stream_raw,
                        res_ptr,
                        ntt_ptr as *const i32,
                        n as i32,
                        min_size as i32,
                        primes_ptr,
                    );
                }
            }

            // Zero extra limbs.
            let extra = res_size.saturating_sub(min_size);
            if extra > 0 {
                let zero_ptr = buf_device_ptr(res_buf, res_col_offset + min_size * cnv_r_limb_bytes(n)) as *mut std::ffi::c_void;
                unsafe {
                    cudaMemsetAsync(zero_ptr, 0, extra * cnv_r_limb_bytes(n), stream_raw);
                }
            }
        }

        stream.synchronize().expect("CUDA sync failed in cnv_prepare_right");
        drop(mask_tmps);
        drop(ntt_tmp);
    }

    // ── cnv_prepare_self ─────────────────────────────────────────────────────
    //
    // Fills both left (CnvPVecL) and right (CnvPVecR) from the same VecZnx,
    // sharing the NTT computation: left ← NTT(a), right ← c_from_b(left).

    fn cnv_prepare_self<'s, 'l, 'r>(
        module: &Module<CudaNtt120Backend>,
        left: &mut CnvPVecLBackendMut<'l, CudaNtt120Backend>,
        right: &mut CnvPVecRBackendMut<'r, CudaNtt120Backend>,
        a: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        mask: i64,
        scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        // Fill left via prepare_left, then convert left → right via c_from_b.
        <CudaNtt120Backend as HalConvolutionImpl<CudaNtt120Backend>>::cnv_prepare_left(module, left, a, mask, scratch);

        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        let primes_ptr = slice_device_ptr(&handle.primes) as *const u32;

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let left_size = left.size();
        let min_size = left_size.min(a.size());
        let cols = left.cols();

        let left_data = left.data();
        let left_buf: &CudaBuf = unsafe { left_data.ptr.as_ref() };

        let right_size = right.size();
        let right_data = right.data_mut();
        let right_buf: &CudaBuf = unsafe { right_data.ptr.as_ref() };

        for col in 0..cols {
            let left_col_offset = left_data.offset + col * left_size * cnv_l_limb_bytes(n);
            let right_col_offset = right_data.offset + col * right_size * cnv_r_limb_bytes(n);

            if min_size > 0 {
                let left_ptr = buf_device_ptr(left_buf, left_col_offset) as *const i32;
                let right_ptr = buf_device_ptr(right_buf, right_col_offset) as *mut u32;
                unsafe {
                    ntt120_svp_c_from_b(stream_raw, right_ptr, left_ptr, n as i32, min_size as i32, primes_ptr);
                }
            }

            // Zero extra limbs in right.
            let extra = right_size.saturating_sub(min_size);
            if extra > 0 {
                let zero_ptr =
                    buf_device_ptr(right_buf, right_col_offset + min_size * cnv_r_limb_bytes(n)) as *mut std::ffi::c_void;
                unsafe {
                    cudaMemsetAsync(zero_ptr, 0, extra * cnv_r_limb_bytes(n), stream_raw);
                }
            }
        }
        // Flush c_from_b kernels before returning (prepare is synchronous).
        stream.synchronize().expect("CUDA sync failed in cnv_prepare_self");
    }

    // ── cnv_apply_dft ────────────────────────────────────────────────────────
    //
    // NTT-domain bbc convolution: res[res_col, k] = Σ_j a[a_col, k_abs-j] ⊙ b[b_col, j].

    fn cnv_apply_dft<'s>(
        module: &Module<CudaNtt120Backend>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, CudaNtt120Backend>,
        b_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        let primes_ptr = slice_device_ptr(&handle.primes) as *const u32;

        let a_size = a.size();
        let b_size = b.size();
        let res_size = res.size();

        let (active_ovs, kernel_offset) = conv_active_ovs(cnv_offset, a_size, b_size, res_size);

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let dft_lb = cnv_l_limb_bytes(n);
        let pvec_r_lb = cnv_r_limb_bytes(n);

        let a_data = a.data();
        let a_buf: &CudaBuf = unsafe { a_data.ptr.as_ref() };
        let a_ptr = buf_device_ptr(a_buf, a_data.offset + a_col * a_size * dft_lb) as *const i32;

        let b_data = b.data();
        let b_buf: &CudaBuf = unsafe { b_data.ptr.as_ref() };
        let b_ptr = buf_device_ptr(b_buf, b_data.offset + b_col * b_size * pvec_r_lb) as *const u32;

        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let res_col_off = res.data.offset + res_col * res_size * dft_lb;
        let res_ptr = buf_device_ptr(res_buf, res_col_off) as *mut i32;

        if active_ovs > 0 {
            unsafe {
                ntt120_cnv_apply_dft(
                    stream_raw,
                    res_ptr,
                    a_ptr,
                    b_ptr,
                    n as i32,
                    a_size as i32,
                    b_size as i32,
                    active_ovs as i32,
                    kernel_offset as i32,
                    primes_ptr,
                );
            }
        }

        let extra = res_size.saturating_sub(active_ovs);
        if extra > 0 {
            let zero_ptr = buf_device_ptr(res_buf, res_col_off + active_ovs * dft_lb) as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, extra * dft_lb, stream_raw);
            }
        }
    }

    // ── cnv_pairwise_apply_dft ───────────────────────────────────────────────
    //
    // Pairwise: res = (a[:,i] + a[:,j]) ⊙ (b[:,i] + b[:,j]).
    // When i == j, delegates to cnv_apply_dft.

    fn cnv_pairwise_apply_dft<'s>(
        module: &Module<CudaNtt120Backend>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, CudaNtt120Backend>,
        b: &CnvPVecRBackendRef<'_, CudaNtt120Backend>,
        col_i: usize,
        col_j: usize,
        scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        if col_i == col_j {
            <CudaNtt120Backend as HalConvolutionImpl<CudaNtt120Backend>>::cnv_apply_dft(
                module, cnv_offset, res, res_col, a, col_i, b, col_j, scratch,
            );
            return;
        }

        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        let primes_ptr = slice_device_ptr(&handle.primes) as *const u32;

        let a_size = a.size();
        let b_size = b.size();
        let res_size = res.size();

        let (active_ovs, kernel_offset) = conv_active_ovs(cnv_offset, a_size, b_size, res_size);

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let dft_lb = cnv_l_limb_bytes(n);
        let pvec_r_lb = cnv_r_limb_bytes(n);

        let a_data = a.data();
        let a_buf: &CudaBuf = unsafe { a_data.ptr.as_ref() };
        let a_i_ptr = buf_device_ptr(a_buf, a_data.offset + col_i * a_size * dft_lb) as *const i32;
        let a_j_ptr = buf_device_ptr(a_buf, a_data.offset + col_j * a_size * dft_lb) as *const i32;

        let b_data = b.data();
        let b_buf: &CudaBuf = unsafe { b_data.ptr.as_ref() };
        let b_i_ptr = buf_device_ptr(b_buf, b_data.offset + col_i * b_size * pvec_r_lb) as *const u32;
        let b_j_ptr = buf_device_ptr(b_buf, b_data.offset + col_j * b_size * pvec_r_lb) as *const u32;

        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let res_col_off = res.data.offset + res_col * res_size * dft_lb;
        let res_ptr = buf_device_ptr(res_buf, res_col_off) as *mut i32;

        if active_ovs > 0 {
            unsafe {
                ntt120_cnv_pairwise_apply_dft(
                    stream_raw,
                    res_ptr,
                    a_i_ptr,
                    a_j_ptr,
                    b_i_ptr,
                    b_j_ptr,
                    n as i32,
                    a_size as i32,
                    b_size as i32,
                    active_ovs as i32,
                    kernel_offset as i32,
                    primes_ptr,
                );
            }
        }

        let extra = res_size.saturating_sub(active_ovs);
        if extra > 0 {
            let zero_ptr = buf_device_ptr(res_buf, res_col_off + active_ovs * dft_lb) as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, extra * dft_lb, stream_raw);
            }
        }
    }

    // ── cnv_by_const_apply ───────────────────────────────────────────────────
    //
    // Coefficient-domain negacyclic inner product into VecZnxBig.
    // b is a host slice that is uploaded to device for the kernel call.

    fn cnv_by_const_apply<'s>(
        module: &Module<CudaNtt120Backend>,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        let n = module.n();

        let a_size = a.size();
        let b_size = b.size();
        let res_size = res.size();

        let (active_ovs, kernel_offset) = conv_active_ovs(cnv_offset, a_size, b_size, res_size);

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let znx_lb = vec_znx_limb_bytes(n);
        let big_lb = vec_znx_big_limb_bytes(n);

        let a_data = a.data;
        let a_buf: &CudaBuf = unsafe { a_data.ptr.as_ref() };
        let a_ptr = buf_device_ptr(a_buf, a_data.offset + a_col * a_size * znx_lb) as *const i64;

        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let res_col_off = res.data.offset + res_col * res_size * big_lb;
        let res_ptr = buf_device_ptr(res_buf, res_col_off) as *mut u32;

        let (mut b_tmp, _) = scratch.borrow().take_vec_znx_scratch(1, 1, b_size);
        module.vec_znx_extract_coeff_backend(&mut b_tmp, 0, b, b_col, b_coeff);
        let b_buf: &CudaBuf = unsafe { b_tmp.data.ptr.as_ref() };
        let b_ptr = buf_device_ptr(b_buf, b_tmp.data.offset) as *const i64;

        if active_ovs > 0 {
            unsafe {
                ntt120_cnv_by_const_apply(
                    stream_raw,
                    res_ptr,
                    a_ptr,
                    b_ptr,
                    n as i32,
                    a_size as i32,
                    b_size as i32,
                    active_ovs as i32,
                    kernel_offset as i32,
                );
            }
        }

        let extra = res_size.saturating_sub(active_ovs);
        if extra > 0 {
            let zero_ptr = buf_device_ptr(res_buf, res_col_off + active_ovs * big_lb) as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, extra * big_lb, stream_raw);
            }
        }

        stream.synchronize().expect("CUDA sync failed in cnv_by_const_apply");
    }
}

// ── Convolution range helper ──────────────────────────────────────────────────

/// Compute (active_ovs, kernel_offset) for a convolution with the given parameters.
///
/// Returns (0, 0) when there is nothing to compute.
#[inline]
fn conv_active_ovs(cnv_offset: usize, a_size: usize, b_size: usize, res_size: usize) -> (usize, usize) {
    if a_size == 0 || b_size == 0 || res_size == 0 {
        return (0, 0);
    }
    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let active_ovs = res_size.min(bound + 1 - offset);
    (active_ovs, offset)
}
