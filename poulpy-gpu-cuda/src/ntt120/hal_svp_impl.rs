//! SVP (scalar-vector product) HAL implementation for `CudaNtt120Backend`.

use std::mem::size_of;

use poulpy_hal::{
    layouts::{
        Backend, Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef,
    },
    oep::HalSvpImpl,
};

use super::hal_impl::{buf_device_ptr, slice_device_ptr};
use crate::ntt120::CudaNtt120Backend;
use crate::{CudaBuf, cuda_stream};

// ── extern "C" kernel launchers ──────────────────────────────────────────────

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

    // src: [cols × 4 × n] i32 NTT output (prime-major).
    // dst: SvpPPol [cols × 4 × n × 2] u32 (prime-major, 2 Montgomery lanes).
    fn ntt120_svp_c_from_b(stream: *mut std::ffi::c_void, dst: *mut u32, src: *const i32, n: i32, cols: i32, primes: *const u32);

    // res[i] = b[i] * a_lane0[i] mod prime[k].  res may alias b (in-place).
    fn ntt120_svp_mul_dft(
        stream: *mut std::ffi::c_void,
        res: *mut i32,
        a: *const u32,
        b: *const i32,
        n: i32,
        size: i32,
        primes: *const u32,
    );

    fn cudaMemsetAsync(
        dev_ptr: *mut std::ffi::c_void,
        value: std::ffi::c_int,
        count: usize,
        stream: *mut std::ffi::c_void,
    ) -> std::ffi::c_int;

    fn cudaMemcpyAsync(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: std::ffi::c_int,
        stream: *mut std::ffi::c_void,
    ) -> std::ffi::c_int;
}

// ── HalSvpImpl ───────────────────────────────────────────────────────────────

unsafe impl HalSvpImpl<CudaNtt120Backend> for CudaNtt120Backend {
    /// Convert one `ScalarZnx` column to the two-lane Montgomery q120c format.
    ///
    /// Allocates a temporary `[4 × n]` i32 device buffer, runs one forward NTT
    /// on `a[a_col]`, converts the residues to SvpPPol lane format via
    /// `ntt120_svp_c_from_b`, then synchronizes and frees the temp buffer.
    fn svp_prepare(
        module: &Module<CudaNtt120Backend>,
        res: &mut SvpPPolBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        let log_n = handle.log_n as i32;

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        // Source: ScalarZnx [cols][n] i64, select column a_col.
        let src_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let src_ptr = buf_device_ptr(src_buf, a.data.offset + a_col * n * size_of::<i64>()) as *const i64;

        // Temp device buffer for NTT output: [4 × n] i32 (one poly, prime-major).
        let tmp_buf = CudaNtt120Backend::alloc_bytes(4 * n * size_of::<i32>());
        let tmp_ptr = tmp_buf.raw_device_ptr(0) as *mut i32;

        let tw_fwd = slice_device_ptr(&handle.twiddle_fwd) as *const u32;
        let tw_fwd_msb = handle
            .twiddle_fwd_msb
            .as_ref()
            .map(|s| slice_device_ptr(s) as *const u32)
            .unwrap_or(std::ptr::null());
        let primes_ptr = slice_device_ptr(&handle.primes) as *const u32;
        let inv_primes_ptr = slice_device_ptr(&handle.inv_primes) as *const i32;

        unsafe {
            ntt120_ntt_fwd_apply(
                stream_raw,
                tmp_ptr,
                src_ptr,
                tw_fwd,
                tw_fwd_msb,
                primes_ptr,
                inv_primes_ptr,
                log_n,
                1,
            );
        }

        // Destination: SvpPPol [cols × 4 × n × 2] u32, column res_col.
        let ppol_col_bytes = n * 4 * 2 * size_of::<u32>();
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let dst_ptr = buf_device_ptr(res_buf, res.data.offset + res_col * ppol_col_bytes) as *mut u32;

        unsafe {
            ntt120_svp_c_from_b(stream_raw, dst_ptr, tmp_ptr as *const i32, n as i32, 1, primes_ptr);
        }

        // Sync before dropping tmp_buf so the GPU finishes consuming it.
        stream.synchronize().expect("CUDA sync failed in svp_prepare");
        drop(tmp_buf);
    }

    /// Device-to-device copy of one `SvpPPol` column.
    fn svp_ppol_copy_backend(
        module: &Module<CudaNtt120Backend>,
        res: &mut SvpPPolBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let col_bytes = n * 4 * 2 * size_of::<u32>();
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let src = buf_device_ptr(a_buf, a.data.offset + a_col * col_bytes) as *const std::ffi::c_void;
        let dst = buf_device_ptr(res_buf, res.data.offset + res_col * col_bytes) as *mut std::ffi::c_void;

        unsafe {
            cudaMemcpyAsync(dst, src, col_bytes, 3 /*DeviceToDevice*/, stream_raw);
        }
    }

    /// Forward NTT `b[b_col]` into `res[res_col]`, then multiply by `a[a_col]` in-place.
    fn svp_apply_dft(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        b_col: usize,
    ) {
        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        let log_n = handle.log_n as i32;
        let min_size = res.size().min(b.size());
        if min_size == 0 {
            return;
        }

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let dft_limb_bytes = 4 * n * size_of::<i32>();

        // VecZnx source: [col][limb][n] i64, column b_col.
        let b_buf: &CudaBuf = unsafe { b.data.ptr.as_ref() };
        let b_ptr = buf_device_ptr(b_buf, b.data.offset + b_col * b.size() * n * size_of::<i64>()) as *const i64;

        // VecZnxDft destination: [col][limb][4][n] i32, column res_col.
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let res_ptr = buf_device_ptr(res_buf, res.data.offset + res_col * res.size() * dft_limb_bytes) as *mut i32;

        let tw_fwd = slice_device_ptr(&handle.twiddle_fwd) as *const u32;
        let tw_fwd_msb = handle
            .twiddle_fwd_msb
            .as_ref()
            .map(|s| slice_device_ptr(s) as *const u32)
            .unwrap_or(std::ptr::null());
        let primes_ptr = slice_device_ptr(&handle.primes) as *const u32;
        let inv_primes_ptr = slice_device_ptr(&handle.inv_primes) as *const i32;

        // Step 1: forward NTT b → res (min_size limbs).
        unsafe {
            ntt120_ntt_fwd_apply(
                stream_raw,
                res_ptr,
                b_ptr,
                tw_fwd,
                tw_fwd_msb,
                primes_ptr,
                inv_primes_ptr,
                log_n,
                min_size as i32,
            );
        }

        // SvpPPol: [col][4 × n × 2] u32, column a_col.
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let a_ptr = buf_device_ptr(a_buf, a.data.offset + a_col * n * 4 * 2 * size_of::<u32>()) as *const u32;

        // Step 2: pointwise multiply res in-place by a.
        unsafe {
            ntt120_svp_mul_dft(
                stream_raw,
                res_ptr,
                a_ptr,
                res_ptr as *const i32,
                n as i32,
                min_size as i32,
                primes_ptr,
            );
        }

        // Step 3: zero any trailing limbs in res that had no corresponding input.
        let extra = res.size().saturating_sub(b.size());
        if extra > 0 {
            let zero_offset = res.data.offset + (res_col * res.size() + min_size) * dft_limb_bytes;
            let zero_ptr = buf_device_ptr(res_buf, zero_offset) as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, extra * dft_limb_bytes, stream_raw);
            }
        }
    }

    /// Pointwise multiply `b[b_col]` by `a[a_col]`, writing the result to `res[res_col]`.
    fn svp_apply_dft_to_dft(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, CudaNtt120Backend>,
        b_col: usize,
    ) {
        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        let min_size = res.size().min(b.size());

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let dft_limb_bytes = 4 * n * size_of::<i32>();
        let primes_ptr = slice_device_ptr(&handle.primes) as *const u32;

        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let b_buf: &CudaBuf = unsafe { b.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };

        let res_ptr = buf_device_ptr(res_buf, res.data.offset + res_col * res.size() * dft_limb_bytes) as *mut i32;
        let b_ptr = buf_device_ptr(b_buf, b.data.offset + b_col * b.size() * dft_limb_bytes) as *const i32;
        let a_ptr = buf_device_ptr(a_buf, a.data.offset + a_col * n * 4 * 2 * size_of::<u32>()) as *const u32;

        if min_size > 0 {
            unsafe {
                ntt120_svp_mul_dft(stream_raw, res_ptr, a_ptr, b_ptr, n as i32, min_size as i32, primes_ptr);
            }
        }

        let extra = res.size().saturating_sub(b.size());
        if extra > 0 {
            let zero_offset = res.data.offset + (res_col * res.size() + min_size) * dft_limb_bytes;
            let zero_ptr = buf_device_ptr(res_buf, zero_offset) as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, extra * dft_limb_bytes, stream_raw);
            }
        }
    }

    /// Pointwise multiply `res[res_col]` in-place by `a[a_col]`.
    fn svp_apply_dft_to_dft_assign(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        if res.size() == 0 {
            return;
        }

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let dft_limb_bytes = 4 * n * size_of::<i32>();
        let primes_ptr = slice_device_ptr(&handle.primes) as *const u32;

        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };

        let res_ptr = buf_device_ptr(res_buf, res.data.offset + res_col * res.size() * dft_limb_bytes) as *mut i32;
        let a_ptr = buf_device_ptr(a_buf, a.data.offset + a_col * n * 4 * 2 * size_of::<u32>()) as *const u32;

        unsafe {
            ntt120_svp_mul_dft(
                stream_raw,
                res_ptr,
                a_ptr,
                res_ptr as *const i32,
                n as i32,
                res.size() as i32,
                primes_ptr,
            );
        }
    }
}
