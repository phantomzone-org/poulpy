//! HAL trait implementations for `CudaNtt120Backend`.

use cudarc::driver::DevicePtr;
use poulpy_hal::{
    layouts::{Module, ScratchArena},
    oep::HalVecZnxDftImpl,
};

use crate::ntt120::CudaNtt120Backend;
use crate::{CudaBuf, cuda_stream};

// ── extern "C" kernel launchers compiled by build.rs ─────────────────────────

unsafe extern "C" {
    fn ntt120_ntt_fwd_apply(
        stream: *mut std::ffi::c_void, // cudaStream_t = CUstream_st*
        dst: *mut i32,
        src: *const i64,
        twiddle_fwd: *const u32,
        twiddle_fwd_msb: *const u32,
        primes: *const u32,
        inv_primes: *const i32,
        log_n: i32,
        batch: i32,
    );

    fn ntt120_ntt_inv_apply(
        stream: *mut std::ffi::c_void,
        dst: *mut i32, // in-place: same buffer as src is valid
        src: *const i32,
        twiddle_inv: *const u32,
        twiddle_inv_msb: *const u32,
        primes: *const u32,
        inv_primes: *const i32,
        inv_n_mont: *const u32,
        log_n: i32,
        batch: i32,
    );

    // CRT: prime-major i32 [cols×4×n] → Big32 [cols×n×4] u32.
    fn ntt120_crt(stream: *mut std::ffi::c_void, dst: *mut u32, src: *const i32, n: i32, cols: i32);

    // DFT-domain pointwise modular add: res[i] = (res[i] + a[i]) mod prime[k].
    fn ntt120_dft_add_assign(stream: *mut std::ffi::c_void, res: *mut i32, a: *const i32, n: i32, size: i32, primes: *const u32);

    // Zero device memory asynchronously (CUDA runtime).
    fn cudaMemsetAsync(
        dev_ptr: *mut std::ffi::c_void,
        value: std::ffi::c_int,
        count: usize,
        stream: *mut std::ffi::c_void,
    ) -> std::ffi::c_int;

    // Device-to-device async copy (cudaMemcpyKind::cudaMemcpyDeviceToDevice = 3).
    fn cudaMemcpyAsync(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: std::ffi::c_int,
        stream: *mut std::ffi::c_void,
    ) -> std::ffi::c_int;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract the raw device pointer (as `u64`) from a `CudaBufRef`/`CudaBufMut`'s
/// underlying `CudaBuf`, applying the view's byte offset.
pub(crate) fn buf_device_ptr(buf: &CudaBuf, offset: usize) -> u64 {
    buf.raw_device_ptr(offset)
}

/// Raw device pointer for a `CudaSlice<T>` stored in the handle.
pub(crate) fn slice_device_ptr<T>(slice: &cudarc::driver::CudaSlice<T>) -> u64 {
    let stream = cuda_stream();
    let (ptr, _guard) = slice.device_ptr(&stream);
    ptr
}

// ── HalVecZnxDftImpl ─────────────────────────────────────────────────────────

unsafe impl HalVecZnxDftImpl<CudaNtt120Backend> for CudaNtt120Backend {
    /// Forward NTT: `VecZnx` i64 → `VecZnxDft` i32 (prime-major).
    ///
    /// Launches Phase1_i64 + Phase2 over all selected limbs in parallel.
    /// `step` must be 1 and `offset` must be 0 (all limbs of the column).
    fn vec_znx_dft_apply(
        module: &Module<CudaNtt120Backend>,
        step: usize,
        offset: usize,
        res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        assert_eq!(step, 1, "CudaNtt120Backend::vec_znx_dft_apply: step != 1 not yet supported");

        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        let log_n = handle.log_n as i32;
        assert!(
            (12..=16).contains(&handle.log_n),
            "CudaNtt120Backend::vec_znx_dft_apply only supports log_n in 12..=16 today; got n={n} (log_n={})",
            handle.log_n
        );

        // Limbs to transform: [offset, a.size()).
        let batch = a.size().saturating_sub(offset) as i32;
        if batch == 0 {
            return;
        }

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        // Source: i64 layout [col][limb][n], select col a_col limb offset..
        let src_byte_offset = a.data.offset + (a_col * a.size() + offset) * n * std::mem::size_of::<i64>();
        let src_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let src_ptr = buf_device_ptr(src_buf, src_byte_offset) as *const i64;

        // Destination: VecZnxDft layout [col][limb][prime][n] i32.
        // bytes_of_vec_znx_dft = 4 * n * cols * size * 4 per the override.
        // Byte offset for (res_col, offset): (res_col * res.size() + offset) * 4 * n * 4.
        let dst_byte_offset = res.data.offset + (res_col * res.size() + offset) * 4 * n * std::mem::size_of::<i32>();
        let dst_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let dst_ptr = buf_device_ptr(dst_buf, dst_byte_offset) as *mut i32;

        // Twiddle pointers from handle.
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
                dst_ptr,
                src_ptr,
                tw_fwd,
                tw_fwd_msb,
                primes_ptr,
                inv_primes_ptr,
                log_n,
                batch,
            );
        }
    }

    fn vec_znx_idft_apply_tmp_bytes(_module: &Module<CudaNtt120Backend>) -> usize {
        0 // tmpa variant is in-place; non-tmpa is unimplemented
    }

    fn vec_znx_idft_apply<'s>(
        _module: &Module<CudaNtt120Backend>,
        _res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        _res_col: usize,
        _a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, CudaNtt120Backend>,
        _a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        unimplemented!("vec_znx_idft_apply: use vec_znx_idft_apply_tmpa instead")
    }

    /// Inverse NTT in-place on `a`'s device buffer, then CRT → `res`.
    ///
    /// INTT Phase1 (9 stages) followed by Phase2 (log_n-9 stages, MultConstNormalize)
    /// are applied in-place in the DFT buffer. Phase1's in-place safety is guaranteed
    /// by the `__syncthreads()` barriers that separate global loads from global stores.
    /// CRT reconstruction then writes Big32 values to `res`.
    fn vec_znx_idft_apply_tmpa(
        module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        let log_n = handle.log_n as i32;

        let batch = a.size() as i32;
        if batch == 0 {
            return;
        }

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        // DFT buffer: [cols × size × 4 × n] i32, prime-major per limb.
        let dft_byte_offset = a.data.offset + a_col * a.size() * 4 * n * std::mem::size_of::<i32>();
        let dft_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let dft_ptr = buf_device_ptr(dft_buf, dft_byte_offset) as *mut i32;

        let tw_inv = slice_device_ptr(&handle.twiddle_inv) as *const u32;
        let tw_inv_msb = handle
            .twiddle_inv_msb
            .as_ref()
            .map(|s| slice_device_ptr(s) as *const u32)
            .unwrap_or(std::ptr::null());
        let primes_ptr = slice_device_ptr(&handle.primes) as *const u32;
        let inv_primes_ptr = slice_device_ptr(&handle.inv_primes) as *const i32;
        let inv_n_mont_ptr = slice_device_ptr(&handle.inv_n_mont) as *const u32;

        unsafe {
            ntt120_ntt_inv_apply(
                stream_raw,
                dft_ptr,
                dft_ptr as *const i32,
                tw_inv,
                tw_inv_msb,
                primes_ptr,
                inv_primes_ptr,
                inv_n_mont_ptr,
                log_n,
                batch,
            );
        }

        // CRT reconstruction: [min_size × 4 × n] i32 → [min_size × n × 4] u32 Big32.
        let min_size = a.size().min(res.size());
        if min_size == 0 {
            return;
        }

        // VecZnxBig layout: [cols × size × n] Big32 = [cols × size × n × 4] u32.
        // size_of::<Big32>() == 16; no bytes_of_vec_znx_big override, so default applies.
        let res_byte_offset = res.data.offset + (res_col * res.size()) * n * std::mem::size_of::<crate::ntt120::types::Big32>();
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let res_ptr = buf_device_ptr(res_buf, res_byte_offset) as *mut u32;

        unsafe {
            ntt120_crt(stream_raw, res_ptr, dft_ptr as *const i32, n as i32, min_size as i32);
        }

        // Zero any trailing limbs in res that have no corresponding input.
        let extra = res.size().saturating_sub(a.size());
        if extra > 0 {
            let zero_byte_offset =
                res.data.offset + (res_col * res.size() + min_size) * n * std::mem::size_of::<crate::ntt120::types::Big32>();
            let zero_ptr = buf_device_ptr(res_buf, zero_byte_offset) as *mut std::ffi::c_void;
            let zero_bytes = extra * n * std::mem::size_of::<crate::ntt120::types::Big32>();
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, zero_bytes, stream_raw);
            }
        }
    }

    fn vec_znx_dft_add_into(
        _module: &Module<CudaNtt120Backend>,
        _res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        _res_col: usize,
        _a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, CudaNtt120Backend>,
        _a_col: usize,
        _b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, CudaNtt120Backend>,
        _b_col: usize,
    ) {
        unimplemented!("vec_znx_dft_add_into: not yet implemented")
    }

    fn vec_znx_dft_add_scaled_assign(
        _module: &Module<CudaNtt120Backend>,
        _res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        _res_col: usize,
        _a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, CudaNtt120Backend>,
        _a_col: usize,
        _a_scale: i64,
    ) {
        unimplemented!("vec_znx_dft_add_scaled_assign: not yet implemented")
    }

    fn vec_znx_dft_add_assign(
        module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        let size = res.size().min(a.size());
        if size == 0 {
            return;
        }

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let res_byte_offset = res.data.offset + res_col * res.size() * 4 * n * std::mem::size_of::<i32>();
        let a_byte_offset = a.data.offset + a_col * a.size() * 4 * n * std::mem::size_of::<i32>();

        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = buf_device_ptr(res_buf, res_byte_offset) as *mut i32;
        let a_ptr = buf_device_ptr(a_buf, a_byte_offset) as *const i32;
        let primes_ptr = slice_device_ptr(&handle.primes) as *const u32;

        unsafe {
            ntt120_dft_add_assign(stream_raw, res_ptr, a_ptr, n as i32, size as i32, primes_ptr);
        }
    }

    fn vec_znx_dft_sub(
        _module: &Module<CudaNtt120Backend>,
        _res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        _res_col: usize,
        _a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, CudaNtt120Backend>,
        _a_col: usize,
        _b: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, CudaNtt120Backend>,
        _b_col: usize,
    ) {
        unimplemented!("vec_znx_dft_sub: not yet implemented")
    }

    fn vec_znx_dft_sub_assign(
        _module: &Module<CudaNtt120Backend>,
        _res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        _res_col: usize,
        _a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, CudaNtt120Backend>,
        _a_col: usize,
    ) {
        unimplemented!("vec_znx_dft_sub_assign: not yet implemented")
    }

    fn vec_znx_dft_sub_negate_assign(
        _module: &Module<CudaNtt120Backend>,
        _res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        _res_col: usize,
        _a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, CudaNtt120Backend>,
        _a_col: usize,
    ) {
        unimplemented!("vec_znx_dft_sub_negate_assign: not yet implemented")
    }

    fn vec_znx_dft_copy(
        module: &Module<CudaNtt120Backend>,
        step: usize,
        offset: usize,
        res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        assert_eq!(step, 1, "CudaNtt120Backend::vec_znx_dft_copy: step != 1 not yet supported");
        assert_eq!(
            offset, 0,
            "CudaNtt120Backend::vec_znx_dft_copy: offset != 0 not yet supported"
        );

        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let elem = 4 * n * std::mem::size_of::<i32>(); // bytes per limb

        let min_size = res.size().min(a.size());
        if min_size > 0 {
            let res_byte_offset = res.data.offset + res_col * res.size() * elem;
            let a_byte_offset = a.data.offset + a_col * a.size() * elem;
            let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
            let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
            let dst = buf_device_ptr(res_buf, res_byte_offset) as *mut std::ffi::c_void;
            let src = buf_device_ptr(a_buf, a_byte_offset) as *const std::ffi::c_void;
            unsafe {
                cudaMemcpyAsync(dst, src, min_size * elem, 3 /*DeviceToDevice*/, stream_raw);
            }
        }

        let extra = res.size().saturating_sub(a.size());
        if extra > 0 {
            let zero_byte_offset = res.data.offset + (res_col * res.size() + min_size) * elem;
            let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
            let ptr = buf_device_ptr(res_buf, zero_byte_offset) as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(ptr, 0, extra * elem, stream_raw);
            }
        }
    }

    fn vec_znx_dft_zero(
        module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
    ) {
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let byte_offset = res.data.offset + res_col * res.size() * 4 * n * std::mem::size_of::<i32>();
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let ptr = buf_device_ptr(res_buf, byte_offset) as *mut std::ffi::c_void;
        let count = res.size() * 4 * n * std::mem::size_of::<i32>();
        unsafe {
            cudaMemsetAsync(ptr, 0, count, stream_raw);
        }
    }
}
