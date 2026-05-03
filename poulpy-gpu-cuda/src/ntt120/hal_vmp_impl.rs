//! VMP (vector-matrix product) HAL implementation for `CudaNtt120Backend`.

use std::mem::size_of;

use poulpy_hal::{
    layouts::{
        Backend, DataView, DataViewMut, MatZnxBackendRef, Module, ScratchArena, VecZnxBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftToBackendMut, VmpPMatBackendMut, VmpPMatBackendRef,
    },
    oep::HalVmpImpl,
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

    fn ntt120_vmp_pmat_pack(
        stream: *mut std::ffi::c_void,
        pmat_dst: *mut u32,
        ntt_src: *const i32,
        n: i32,
        in_rows: i32,
        out_vecs: i32,
    );

    fn ntt120_vmp_apply(
        stream: *mut std::ffi::c_void,
        res: *mut i32,
        a: *const i32,
        pmat: *const u32,
        n: i32,
        a_row_max: i32,
        in_rows: i32,
        out_vecs: i32,
        active_ovs: i32,
        pmat_col_off: i64,
        primes: *const u32,
    );

    fn cudaMemsetAsync(
        dev_ptr: *mut std::ffi::c_void,
        value: std::ffi::c_int,
        count: usize,
        stream: *mut std::ffi::c_void,
    ) -> std::ffi::c_int;
}

// ── HalVmpImpl ───────────────────────────────────────────────────────────────

unsafe impl HalVmpImpl<CudaNtt120Backend> for CudaNtt120Backend {
    fn vmp_prepare_tmp_bytes(
        _module: &Module<CudaNtt120Backend>,
        _rows: usize,
        _cols_in: usize,
        _cols_out: usize,
        _size: usize,
    ) -> usize {
        0
    }

    /// Batch-NTT all polynomials from `a` (a MatZnx) into a temporary device
    /// buffer, then repack into the GPU-native VmpPMat block layout via
    /// `ntt120_vmp_pmat_pack`. Allocates the temp NTT buffer internally and
    /// syncs before dropping it.
    fn vmp_prepare<'s>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VmpPMatBackendMut<'_, CudaNtt120Backend>,
        a: &MatZnxBackendRef<'_, CudaNtt120Backend>,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        let log_n = handle.log_n as i32;

        let in_rows = a.rows() * a.cols_in();
        let out_vecs = a.cols_out() * a.size();
        let batch = in_rows * out_vecs;
        if batch == 0 {
            return;
        }

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let a_data = a.data();
        let a_buf: &CudaBuf = unsafe { a_data.ptr.as_ref() };
        let a_ptr = buf_device_ptr(a_buf, a_data.offset) as *const i64;

        let tmp_buf = CudaNtt120Backend::alloc_bytes(batch * 4 * n * size_of::<i32>());
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
                a_ptr,
                tw_fwd,
                tw_fwd_msb,
                primes_ptr,
                inv_primes_ptr,
                log_n,
                batch as i32,
            );
        }

        let res_data = res.data_mut();
        let res_buf: &CudaBuf = unsafe { res_data.ptr.as_ref() };
        let res_ptr = buf_device_ptr(res_buf, res_data.offset) as *mut u32;

        unsafe {
            ntt120_vmp_pmat_pack(
                stream_raw,
                res_ptr,
                tmp_ptr as *const i32,
                n as i32,
                in_rows as i32,
                out_vecs as i32,
            );
        }

        stream.synchronize().expect("CUDA sync failed in vmp_prepare");
        drop(tmp_buf);
    }

    fn vmp_apply_dft_tmp_bytes(
        _module: &Module<CudaNtt120Backend>,
        _res_size: usize,
        _a_size: usize,
        _b_rows: usize,
        _b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    /// Forward-NTT all limbs of the coefficient-domain `a` into an internal
    /// temp DFT buffer, then apply the VMP kernel against `b`. Syncs and drops
    /// the temp buffer before returning.
    fn vmp_apply_dft<'s, R>(
        module: &Module<CudaNtt120Backend>,
        res: &mut R,
        a: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        b: &VmpPMatBackendRef<'_, CudaNtt120Backend>,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) where
        R: VecZnxDftToBackendMut<CudaNtt120Backend>,
    {
        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        let log_n = handle.log_n as i32;
        let dft_limb_bytes = 4 * n * size_of::<i32>();

        let in_rows = b.rows() * b.cols_in();
        let a_total = a.cols() * a.size();
        let a_row_max = a_total.min(in_rows);

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let res_backend = res.to_backend_mut();
        let res_total = res_backend.cols() * res_backend.size();
        let out_vecs = b.cols_out() * b.size();

        if a_row_max == 0 {
            let res_buf: &CudaBuf = unsafe { res_backend.data.ptr.as_ref() };
            let zero_ptr = buf_device_ptr(res_buf, res_backend.data.offset) as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, res_total * dft_limb_bytes, stream_raw);
            }
            return;
        }

        let tw_fwd = slice_device_ptr(&handle.twiddle_fwd) as *const u32;
        let tw_fwd_msb = handle
            .twiddle_fwd_msb
            .as_ref()
            .map(|s| slice_device_ptr(s) as *const u32)
            .unwrap_or(std::ptr::null());
        let primes_ptr = slice_device_ptr(&handle.primes) as *const u32;
        let inv_primes_ptr = slice_device_ptr(&handle.inv_primes) as *const i32;

        let tmp_buf = CudaNtt120Backend::alloc_bytes(a_row_max * dft_limb_bytes);
        let tmp_ptr = tmp_buf.raw_device_ptr(0) as *mut i32;

        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let a_ptr = buf_device_ptr(a_buf, a.data.offset) as *const i64;

        unsafe {
            ntt120_ntt_fwd_apply(
                stream_raw,
                tmp_ptr,
                a_ptr,
                tw_fwd,
                tw_fwd_msb,
                primes_ptr,
                inv_primes_ptr,
                log_n,
                a_row_max as i32,
            );
        }

        let active_ovs = res_total.min(out_vecs);
        let b_data = b.data();
        let b_buf: &CudaBuf = unsafe { b_data.ptr.as_ref() };
        let b_ptr = buf_device_ptr(b_buf, b_data.offset) as *const u32;
        let res_buf: &CudaBuf = unsafe { res_backend.data.ptr.as_ref() };
        let res_ptr = buf_device_ptr(res_buf, res_backend.data.offset) as *mut i32;

        if active_ovs > 0 {
            unsafe {
                ntt120_vmp_apply(
                    stream_raw,
                    res_ptr,
                    tmp_ptr as *const i32,
                    b_ptr,
                    n as i32,
                    a_row_max as i32,
                    in_rows as i32,
                    out_vecs as i32,
                    active_ovs as i32,
                    0i64,
                    primes_ptr,
                );
            }
        }

        let extra = res_total.saturating_sub(active_ovs);
        if extra > 0 {
            let zero_ptr =
                buf_device_ptr(res_buf, res_backend.data.offset + active_ovs * dft_limb_bytes) as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, extra * dft_limb_bytes, stream_raw);
            }
        }

        stream.synchronize().expect("CUDA sync failed in vmp_apply_dft");
        drop(tmp_buf);
    }

    fn vmp_apply_dft_to_dft_tmp_bytes(
        _module: &Module<CudaNtt120Backend>,
        _res_size: usize,
        _a_size: usize,
        _b_rows: usize,
        _b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        0
    }

    fn vmp_apply_dft_to_dft<'s, R>(
        module: &Module<CudaNtt120Backend>,
        res: &mut R,
        a: &VecZnxDftBackendRef<'_, CudaNtt120Backend>,
        b: &VmpPMatBackendRef<'_, CudaNtt120Backend>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) where
        R: VecZnxDftToBackendMut<CudaNtt120Backend>,
    {
        let mut res_backend = res.to_backend_mut();
        <CudaNtt120Backend as HalVmpImpl<CudaNtt120Backend>>::vmp_apply_dft_to_dft_backend_ref(
            module,
            &mut res_backend,
            a,
            b,
            limb_offset,
            scratch,
        );
    }

    /// Pointwise multiply `a × b[pmat_col_off..]` writing `active_ovs` output
    /// limbs to `res`. Zeros any trailing limbs in `res` that have no matching
    /// pmat column. All operations are async on the per-thread CUDA stream.
    fn vmp_apply_dft_to_dft_backend_ref<'s, 'r, 'aa>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxDftBackendMut<'r, CudaNtt120Backend>,
        a: &VecZnxDftBackendRef<'aa, CudaNtt120Backend>,
        b: &VmpPMatBackendRef<'_, CudaNtt120Backend>,
        limb_offset: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        let handle = unsafe { &*module.as_mut_ptr() };
        let n = module.n();
        let dft_limb_bytes = 4 * n * size_of::<i32>();

        let in_rows = b.rows() * b.cols_in();
        let out_vecs = b.cols_out() * b.size();
        let pmat_col_off = (limb_offset * b.cols_out()) as i64;
        let a_total = a.cols() * a.size();
        let a_row_max = a_total.min(in_rows);
        let res_total = res.cols() * res.size();
        let available = out_vecs.saturating_sub(pmat_col_off as usize);
        let active_ovs = res_total.min(available);

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let primes_ptr = slice_device_ptr(&handle.primes) as *const u32;

        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let b_data = b.data();
        let b_buf: &CudaBuf = unsafe { b_data.ptr.as_ref() };

        let res_ptr = buf_device_ptr(res_buf, res.data.offset) as *mut i32;
        let a_ptr = buf_device_ptr(a_buf, a.data.offset) as *const i32;
        let b_ptr = buf_device_ptr(b_buf, b_data.offset) as *const u32;

        if active_ovs > 0 && a_row_max > 0 {
            unsafe {
                ntt120_vmp_apply(
                    stream_raw,
                    res_ptr,
                    a_ptr,
                    b_ptr,
                    n as i32,
                    a_row_max as i32,
                    in_rows as i32,
                    out_vecs as i32,
                    active_ovs as i32,
                    pmat_col_off,
                    primes_ptr,
                );
            }
        }

        let extra = res_total.saturating_sub(active_ovs);
        if extra > 0 {
            let zero_ptr = buf_device_ptr(res_buf, res.data.offset + active_ovs * dft_limb_bytes) as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, extra * dft_limb_bytes, stream_raw);
            }
        }
    }

    fn vmp_zero(module: &Module<CudaNtt120Backend>, res: &mut VmpPMatBackendMut<'_, CudaNtt120Backend>) {
        let n = module.n();
        let total_bytes = CudaNtt120Backend::bytes_of_vmp_pmat(n, res.rows(), res.cols_in(), res.cols_out(), res.size());

        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let res_data = res.data_mut();
        let res_buf: &CudaBuf = unsafe { res_data.ptr.as_ref() };
        let res_ptr = buf_device_ptr(res_buf, res_data.offset) as *mut std::ffi::c_void;

        unsafe {
            cudaMemsetAsync(res_ptr, 0, total_bytes, stream_raw);
        }
    }
}
