//! VecZnx coefficient-domain HAL implementation for `CudaNtt120Backend`.
//!
//! GPU VecZnx layout is column-major: `[cols][size][n]` i64.
//! Column `col` of a buffer with per-column stride `size * n` starts at
//!   `data.offset + col * size * n * sizeof(i64)` bytes from the buffer base.

use std::mem::size_of;

use poulpy_cpu_ref::reference::znx::{znx_add_normal_f64_ref, znx_fill_normal_f64_ref, znx_fill_uniform_ref};
use poulpy_hal::{
    layouts::{
        Backend, HostBytesBackend, Module, NoiseInfos, ScalarZnx, ScalarZnxBackendMut, ScalarZnxBackendRef, ScratchArena,
        VecZnxBackendMut, VecZnxBackendRef,
    },
    oep::HalVecZnxImpl,
    source::Source,
};

use super::hal_impl::buf_device_ptr;
use crate::ntt120::CudaNtt120Backend;
use crate::{CudaBuf, cuda_stream};

// ── extern "C" kernel launchers ──────────────────────────────────────────────

unsafe extern "C" {
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

    // Synchronous host↔device copy (kind: 1=H2D, 2=D2H).
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: std::ffi::c_int,
    ) -> std::ffi::c_int;

    fn ntt120_vec_znx_add_into(stream: *mut std::ffi::c_void, res: *mut i64, a: *const i64, b: *const i64, len: std::ffi::c_int);

    fn ntt120_vec_znx_add_assign(stream: *mut std::ffi::c_void, res: *mut i64, a: *const i64, len: std::ffi::c_int);

    fn ntt120_vec_znx_sub_into(stream: *mut std::ffi::c_void, res: *mut i64, a: *const i64, b: *const i64, len: std::ffi::c_int);

    fn ntt120_vec_znx_sub_inplace(stream: *mut std::ffi::c_void, res: *mut i64, a: *const i64, len: std::ffi::c_int);

    fn ntt120_vec_znx_sub_negate_inplace(stream: *mut std::ffi::c_void, res: *mut i64, a: *const i64, len: std::ffi::c_int);

    fn ntt120_vec_znx_negate(stream: *mut std::ffi::c_void, res: *mut i64, a: *const i64, len: std::ffi::c_int);

    fn ntt120_vec_znx_negate_inplace(stream: *mut std::ffi::c_void, res: *mut i64, len: std::ffi::c_int);

    fn ntt120_vec_znx_add_scalar_assign(stream: *mut std::ffi::c_void, res: *mut i64, a: *const i64, n: std::ffi::c_int);

    fn ntt120_vec_znx_add_scalar_into(
        stream: *mut std::ffi::c_void,
        res: *mut i64,
        a: *const i64,
        b: *const i64,
        n: std::ffi::c_int,
    );

    fn ntt120_vec_znx_sub_scalar_into(
        stream: *mut std::ffi::c_void,
        res: *mut i64,
        a: *const i64,
        b: *const i64,
        n: std::ffi::c_int,
    );

    fn ntt120_vec_znx_sub_scalar_inplace(stream: *mut std::ffi::c_void, res: *mut i64, a: *const i64, n: std::ffi::c_int);

    fn ntt120_vec_znx_rotate(
        stream: *mut std::ffi::c_void,
        res: *mut i64,
        a: *const i64,
        n: std::ffi::c_int,
        nlimbs: std::ffi::c_int,
        p: i64,
    );

    fn ntt120_vec_znx_mul_xp_minus_one(
        stream: *mut std::ffi::c_void,
        res: *mut i64,
        a: *const i64,
        n: std::ffi::c_int,
        nlimbs: std::ffi::c_int,
        p: i64,
    );

    fn ntt120_vec_znx_automorphism(
        stream: *mut std::ffi::c_void,
        res: *mut i64,
        a: *const i64,
        n: std::ffi::c_int,
        nlimbs: std::ffi::c_int,
        p: i64,
    );

    // Shift and normalize kernels
    fn ntt120_vec_znx_lsh(
        stream: *mut std::ffi::c_void,
        res: *mut i64,
        a: *const i64,
        n: std::ffi::c_int,
        res_size: std::ffi::c_int,
        a_size: std::ffi::c_int,
        base2k: std::ffi::c_int,
        steps: std::ffi::c_int,
        k_rem: std::ffi::c_int,
        mode: std::ffi::c_int,
    );

    fn ntt120_vec_znx_rsh(
        stream: *mut std::ffi::c_void,
        res: *mut i64,
        a: *const i64,
        n: std::ffi::c_int,
        res_size: std::ffi::c_int,
        a_size: std::ffi::c_int,
        base2k: std::ffi::c_int,
        steps: std::ffi::c_int,
        lsh: std::ffi::c_int,
        mode: std::ffi::c_int,
    );

    fn ntt120_vec_znx_normalize(
        stream: *mut std::ffi::c_void,
        res: *mut i64,
        a: *const i64,
        n: std::ffi::c_int,
        res_size: std::ffi::c_int,
        a_size: std::ffi::c_int,
        base2k: std::ffi::c_int,
        res_offset: i64,
    );

    fn ntt120_vec_znx_normalize_inplace(
        stream: *mut std::ffi::c_void,
        a: *mut i64,
        n: std::ffi::c_int,
        size: std::ffi::c_int,
        base2k: std::ffi::c_int,
    );

    fn ntt120_vec_znx_rsh_inplace(
        stream: *mut std::ffi::c_void,
        a: *mut i64,
        n: std::ffi::c_int,
        size: std::ffi::c_int,
        base2k: std::ffi::c_int,
        steps: std::ffi::c_int,
        lsh: std::ffi::c_int,
    );

    fn ntt120_vec_znx_lsh_inplace(
        stream: *mut std::ffi::c_void,
        a: *mut i64,
        n: std::ffi::c_int,
        size: std::ffi::c_int,
        base2k: std::ffi::c_int,
        steps: std::ffi::c_int,
        k_rem: std::ffi::c_int,
    );

    fn ntt120_vec_znx_split_ring(
        stream: *mut std::ffi::c_void,
        res_even: *mut i64,
        res_odd: *mut i64,
        a: *const i64,
        n_full: std::ffi::c_int,
        n_half: std::ffi::c_int,
        nlimbs: std::ffi::c_int,
    );

    fn ntt120_vec_znx_merge_rings(
        stream: *mut std::ffi::c_void,
        res: *mut i64,
        a_even: *const i64,
        a_odd: *const i64,
        n_full: std::ffi::c_int,
        n_half: std::ffi::c_int,
        nlimbs: std::ffi::c_int,
    );

    fn ntt120_vec_znx_switch_ring(
        stream: *mut std::ffi::c_void,
        res: *mut i64,
        a: *const i64,
        n_src: std::ffi::c_int,
        n_dst: std::ffi::c_int,
        nlimbs: std::ffi::c_int,
        direction: std::ffi::c_int,
    );

    fn ntt120_vec_znx_sub_dot(
        stream: *mut std::ffi::c_void,
        res_ptr: *mut i64,
        a: *const i64,
        b: *const i64,
        len: std::ffi::c_int,
    );
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Returns a raw device pointer (as u64) to the first element of column `col`
/// of a VecZnx buffer (GPU column-major layout: `[cols][size][n]` i64).
#[inline]
fn vec_znx_col_ptr(buf: &CudaBuf, offset: usize, col: usize, size: usize, n: usize) -> u64 {
    buf_device_ptr(buf, offset + col * size * n * size_of::<i64>())
}

/// Returns the raw device pointer (as u64) to column `col` of a
/// `ScalarZnx` buffer (GPU layout: `[cols][n]` i64).
#[inline]
fn scalar_znx_col_ptr(buf: &CudaBuf, offset: usize, col: usize, n: usize) -> u64 {
    buf_device_ptr(buf, offset + col * n * size_of::<i64>())
}

#[inline]
fn alloc_host_scalar_znx(n: usize, cols: usize) -> ScalarZnx<Vec<u8>> {
    ScalarZnx::from_data(
        HostBytesBackend::alloc_bytes(ScalarZnx::<Vec<u8>>::bytes_of(n, cols)),
        n,
        cols,
    )
}

// ── HalVecZnxImpl ────────────────────────────────────────────────────────────

unsafe impl HalVecZnxImpl<CudaNtt120Backend> for CudaNtt120Backend {
    // ── scalar_znx sampling — CPU generation + H2D upload ────────────────────

    fn scalar_znx_fill_ternary_hw_backend(
        module: &Module<CudaNtt120Backend>,
        res: &mut ScalarZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        hw: usize,
        seed: [u8; 32],
    ) {
        let n = module.n();
        let mut source = Source::new(seed);
        let mut cpu = alloc_host_scalar_znx(n, 1);
        cpu.fill_ternary_hw(0, hw, &mut source);
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let dev_ptr = scalar_znx_col_ptr(res_buf, res.data.offset, res_col, n) as *mut std::ffi::c_void;
        unsafe {
            cudaMemcpy(dev_ptr, cpu.data.as_ptr() as _, n * size_of::<i64>(), 1 /*H2D*/);
        }
    }

    fn scalar_znx_fill_ternary_prob_backend(
        module: &Module<CudaNtt120Backend>,
        res: &mut ScalarZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        prob: f64,
        seed: [u8; 32],
    ) {
        let n = module.n();
        let mut source = Source::new(seed);
        let mut cpu = alloc_host_scalar_znx(n, 1);
        cpu.fill_ternary_prob(0, prob, &mut source);
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let dev_ptr = scalar_znx_col_ptr(res_buf, res.data.offset, res_col, n) as *mut std::ffi::c_void;
        unsafe {
            cudaMemcpy(dev_ptr, cpu.data.as_ptr() as _, n * size_of::<i64>(), 1 /*H2D*/);
        }
    }

    fn scalar_znx_fill_binary_hw_backend(
        module: &Module<CudaNtt120Backend>,
        res: &mut ScalarZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        hw: usize,
        seed: [u8; 32],
    ) {
        let n = module.n();
        let mut source = Source::new(seed);
        let mut cpu = alloc_host_scalar_znx(n, 1);
        cpu.fill_binary_hw(0, hw, &mut source);
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let dev_ptr = scalar_znx_col_ptr(res_buf, res.data.offset, res_col, n) as *mut std::ffi::c_void;
        unsafe {
            cudaMemcpy(dev_ptr, cpu.data.as_ptr() as _, n * size_of::<i64>(), 1 /*H2D*/);
        }
    }

    fn scalar_znx_fill_binary_prob_backend(
        module: &Module<CudaNtt120Backend>,
        res: &mut ScalarZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        prob: f64,
        seed: [u8; 32],
    ) {
        let n = module.n();
        let mut source = Source::new(seed);
        let mut cpu = alloc_host_scalar_znx(n, 1);
        cpu.fill_binary_prob(0, prob, &mut source);
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let dev_ptr = scalar_znx_col_ptr(res_buf, res.data.offset, res_col, n) as *mut std::ffi::c_void;
        unsafe {
            cudaMemcpy(dev_ptr, cpu.data.as_ptr() as _, n * size_of::<i64>(), 1 /*H2D*/);
        }
    }

    fn scalar_znx_fill_binary_block_backend(
        module: &Module<CudaNtt120Backend>,
        res: &mut ScalarZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        block_size: usize,
        seed: [u8; 32],
    ) {
        let n = module.n();
        let mut source = Source::new(seed);
        let mut cpu = alloc_host_scalar_znx(n, 1);
        cpu.fill_binary_block(0, block_size, &mut source);
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let dev_ptr = scalar_znx_col_ptr(res_buf, res.data.offset, res_col, n) as *mut std::ffi::c_void;
        unsafe {
            cudaMemcpy(dev_ptr, cpu.data.as_ptr() as _, n * size_of::<i64>(), 1 /*H2D*/);
        }
    }

    // ── vec_znx_zero ─────────────────────────────────────────────────────────

    fn vec_znx_zero_backend<'r>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
    ) {
        if res.size() == 0 {
            return;
        }
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut std::ffi::c_void;
        unsafe {
            cudaMemsetAsync(ptr, 0, res.size() * n * size_of::<i64>(), stream_raw);
        }
    }

    // ── vec_znx inner-product ────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_inner_product_assign_backend<'r, 'a, 'b>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        res_limb: usize,
        res_offset: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
        a_limb: usize,
        a_offset: usize,
        b: &ScalarZnxBackendRef<'b, CudaNtt120Backend>,
        b_col: usize,
        b_offset: usize,
        len: usize,
    ) {
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let b_buf: &CudaBuf = unsafe { b.data.ptr.as_ref() };
        // res[res_col][res_limb][res_offset]
        let res_ptr = buf_device_ptr(
            res_buf,
            res.data.offset + ((res_col * res.size() + res_limb) * n + res_offset) * size_of::<i64>(),
        ) as *mut i64;
        // a[a_col][a_limb][a_offset]
        let a_ptr = buf_device_ptr(
            a_buf,
            a.data.offset + ((a_col * a.size() + a_limb) * n + a_offset) * size_of::<i64>(),
        ) as *const i64;
        // b[b_col][0][b_offset]  (ScalarZnx has one limb per column: layout [cols][n])
        let b_ptr = buf_device_ptr(b_buf, b.data.offset + (b_col * n + b_offset) * size_of::<i64>()) as *const i64;
        unsafe {
            ntt120_vec_znx_sub_dot(stream_raw, res_ptr, a_ptr, b_ptr, len as std::ffi::c_int);
        }
    }

    // ── normalize ────────────────────────────────────────────────────────────

    fn vec_znx_normalize_tmp_bytes(_module: &Module<CudaNtt120Backend>) -> usize {
        0
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize<'s, 'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_base2k: usize,
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        assert_eq!(
            res_base2k, a_base2k,
            "CudaNtt120Backend::vec_znx_normalize: cross-base2k not implemented"
        );
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        unsafe {
            ntt120_vec_znx_normalize(
                stream_raw,
                res_ptr,
                a_ptr,
                n as i32,
                res.size() as i32,
                a.size() as i32,
                res_base2k as i32,
                res_offset,
            );
        }
    }

    fn vec_znx_normalize_inplace_backend<'s, 'r>(
        module: &Module<CudaNtt120Backend>,
        base2k: usize,
        a: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        if a.size() == 0 {
            return;
        }
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *mut i64;
        unsafe {
            ntt120_vec_znx_normalize_inplace(stream_raw, a_ptr, n as i32, a.size() as i32, base2k as i32);
        }
    }

    // ── add / sub ────────────────────────────────────────────────────────────

    fn vec_znx_add_into_backend<'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        b_col: usize,
    ) {
        let n = module.n();
        let min_size = res.size().min(a.size()).min(b.size());
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let b_buf: &CudaBuf = unsafe { b.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        let b_ptr = vec_znx_col_ptr(b_buf, b.data.offset, b_col, b.size(), n) as *const i64;
        if min_size > 0 {
            unsafe {
                ntt120_vec_znx_add_into(stream_raw, res_ptr, a_ptr, b_ptr, (min_size * n) as i32);
            }
        }
        if res.size() > min_size {
            let zero_ptr = unsafe { res_ptr.add(min_size * n) } as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, (res.size() - min_size) * n * size_of::<i64>(), stream_raw);
            }
        }
    }

    fn vec_znx_add_assign_backend<'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = module.n();
        let min_size = res.size().min(a.size());
        if min_size == 0 {
            return;
        }
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        unsafe {
            ntt120_vec_znx_add_assign(stream_raw, res_ptr, a_ptr, (min_size * n) as i32);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_scalar_into_backend<'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        b_col: usize,
        b_limb: usize,
    ) {
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let b_buf: &CudaBuf = unsafe { b.data.ptr.as_ref() };
        // res[res_col][0] = a[a_col] + b[b_col][b_limb]
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = scalar_znx_col_ptr(a_buf, a.data.offset, a_col, n) as *const i64;
        // b's limb b_limb within column b_col
        let b_limb_offset = b.data.offset + (b_col * b.size() + b_limb) * n * size_of::<i64>();
        let b_ptr = buf_device_ptr(b_buf, b_limb_offset) as *const i64;
        unsafe {
            ntt120_vec_znx_add_scalar_into(stream_raw, res_ptr, a_ptr, b_ptr, n as i32);
        }
    }

    fn vec_znx_add_scalar_assign_backend<'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_limb_offset = res.data.offset + (res_col * res.size() + res_limb) * n * size_of::<i64>();
        let res_ptr = buf_device_ptr(res_buf, res_limb_offset) as *mut i64;
        let a_ptr = scalar_znx_col_ptr(a_buf, a.data.offset, a_col, n) as *const i64;
        unsafe {
            ntt120_vec_znx_add_scalar_assign(stream_raw, res_ptr, a_ptr, n as i32);
        }
    }

    fn vec_znx_sub_backend<'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        b_col: usize,
    ) {
        let n = module.n();
        let min_size = res.size().min(a.size()).min(b.size());
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let b_buf: &CudaBuf = unsafe { b.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        let b_ptr = vec_znx_col_ptr(b_buf, b.data.offset, b_col, b.size(), n) as *const i64;
        if min_size > 0 {
            unsafe {
                ntt120_vec_znx_sub_into(stream_raw, res_ptr, a_ptr, b_ptr, (min_size * n) as i32);
            }
        }
        if res.size() > min_size {
            let zero_ptr = unsafe { res_ptr.add(min_size * n) } as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, (res.size() - min_size) * n * size_of::<i64>(), stream_raw);
            }
        }
    }

    fn vec_znx_sub_inplace_backend<'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = module.n();
        let min_size = res.size().min(a.size());
        if min_size == 0 {
            return;
        }
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        unsafe {
            ntt120_vec_znx_sub_inplace(stream_raw, res_ptr, a_ptr, (min_size * n) as i32);
        }
    }

    fn vec_znx_sub_negate_inplace_backend<'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = module.n();
        let min_size = res.size().min(a.size());
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        if min_size > 0 {
            unsafe {
                ntt120_vec_znx_sub_negate_inplace(stream_raw, res_ptr, a_ptr, (min_size * n) as i32);
            }
        }
        // limbs in res beyond min_size are negated (res = a - res; a=0 there means res = 0 - res = -res)
        if res.size() > min_size {
            let extra_ptr = unsafe { res_ptr.add(min_size * n) };
            unsafe {
                ntt120_vec_znx_negate_inplace(stream_raw, extra_ptr, ((res.size() - min_size) * n) as i32);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_scalar_backend<'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        b_col: usize,
        b_limb: usize,
    ) {
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let b_buf: &CudaBuf = unsafe { b.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = scalar_znx_col_ptr(a_buf, a.data.offset, a_col, n) as *const i64;
        let b_limb_offset = b.data.offset + (b_col * b.size() + b_limb) * n * size_of::<i64>();
        let b_ptr = buf_device_ptr(b_buf, b_limb_offset) as *const i64;
        unsafe {
            ntt120_vec_znx_sub_scalar_into(stream_raw, res_ptr, a_ptr, b_ptr, n as i32);
        }
    }

    fn vec_znx_sub_scalar_inplace_backend<'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_limb_offset = res.data.offset + (res_col * res.size() + res_limb) * n * size_of::<i64>();
        let res_ptr = buf_device_ptr(res_buf, res_limb_offset) as *mut i64;
        let a_ptr = scalar_znx_col_ptr(a_buf, a.data.offset, a_col, n) as *const i64;
        unsafe {
            ntt120_vec_znx_sub_scalar_inplace(stream_raw, res_ptr, a_ptr, n as i32);
        }
    }

    // ── negate ───────────────────────────────────────────────────────────────

    fn vec_znx_negate_backend(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = module.n();
        let min_size = res.size().min(a.size());
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        if min_size > 0 {
            unsafe {
                ntt120_vec_znx_negate(stream_raw, res_ptr, a_ptr, (min_size * n) as i32);
            }
        }
        if res.size() > min_size {
            let zero_ptr = unsafe { res_ptr.add(min_size * n) } as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, (res.size() - min_size) * n * size_of::<i64>(), stream_raw);
            }
        }
    }

    fn vec_znx_negate_inplace_backend(
        module: &Module<CudaNtt120Backend>,
        a: &mut VecZnxBackendMut<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        if a.size() == 0 {
            return;
        }
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *mut i64;
        unsafe {
            ntt120_vec_znx_negate_inplace(stream_raw, a_ptr, (a.size() * n) as i32);
        }
    }

    // ── rsh / lsh (not yet GPU-native) ───────────────────────────────────────

    fn vec_znx_rsh_tmp_bytes(_module: &Module<CudaNtt120Backend>) -> usize {
        0
    }

    fn vec_znx_rsh_backend<'s, 'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        let n = module.n();
        let mut steps = k / base2k;
        let k_rem = k % base2k;
        if k_rem != 0 {
            steps += 1;
        }
        let lsh = if k_rem != 0 { base2k - k_rem } else { 0 };
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        unsafe {
            ntt120_vec_znx_rsh(
                stream_raw,
                res_ptr,
                a_ptr,
                n as i32,
                res.size() as i32,
                a.size() as i32,
                base2k as i32,
                steps as i32,
                lsh as i32,
                0,
            );
        }
    }

    fn vec_znx_rsh_add_into_backend<'s, 'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        let n = module.n();
        let mut steps = k / base2k;
        let k_rem = k % base2k;
        if k_rem != 0 {
            steps += 1;
        }
        let lsh = if k_rem != 0 { base2k - k_rem } else { 0 };
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        unsafe {
            ntt120_vec_znx_rsh(
                stream_raw,
                res_ptr,
                a_ptr,
                n as i32,
                res.size() as i32,
                a.size() as i32,
                base2k as i32,
                steps as i32,
                lsh as i32,
                1,
            );
        }
    }

    fn vec_znx_lsh_tmp_bytes(_module: &Module<CudaNtt120Backend>) -> usize {
        0
    }

    fn vec_znx_lsh_backend<'s, 'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        let n = module.n();
        let steps = k / base2k;
        let k_rem = k % base2k;
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        unsafe {
            ntt120_vec_znx_lsh(
                stream_raw,
                res_ptr,
                a_ptr,
                n as i32,
                res.size() as i32,
                a.size() as i32,
                base2k as i32,
                steps as i32,
                k_rem as i32,
                0,
            );
        }
    }

    fn vec_znx_lsh_add_into_backend<'s, 'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        let n = module.n();
        let steps = k / base2k;
        let k_rem = k % base2k;
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        unsafe {
            ntt120_vec_znx_lsh(
                stream_raw,
                res_ptr,
                a_ptr,
                n as i32,
                res.size() as i32,
                a.size() as i32,
                base2k as i32,
                steps as i32,
                k_rem as i32,
                1,
            );
        }
    }

    fn vec_znx_lsh_sub_backend<'s, 'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        let n = module.n();
        let steps = k / base2k;
        let k_rem = k % base2k;
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        unsafe {
            ntt120_vec_znx_lsh(
                stream_raw,
                res_ptr,
                a_ptr,
                n as i32,
                res.size() as i32,
                a.size() as i32,
                base2k as i32,
                steps as i32,
                k_rem as i32,
                2,
            );
        }
    }

    fn vec_znx_rsh_sub_backend<'s, 'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        let n = module.n();
        let mut steps = k / base2k;
        let k_rem = k % base2k;
        if k_rem != 0 {
            steps += 1;
        }
        let lsh = if k_rem != 0 { base2k - k_rem } else { 0 };
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        unsafe {
            ntt120_vec_znx_rsh(
                stream_raw,
                res_ptr,
                a_ptr,
                n as i32,
                res.size() as i32,
                a.size() as i32,
                base2k as i32,
                steps as i32,
                lsh as i32,
                2,
            );
        }
    }

    fn vec_znx_rsh_inplace_backend<'s, 'r>(
        module: &Module<CudaNtt120Backend>,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        if a.size() == 0 {
            return;
        }
        let n = module.n();
        let mut steps = k / base2k;
        let k_rem = k % base2k;
        if k_rem != 0 {
            steps += 1;
        }
        let lsh = if k_rem != 0 { base2k - k_rem } else { 0 };
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *mut i64;
        unsafe {
            ntt120_vec_znx_rsh_inplace(
                stream_raw,
                a_ptr,
                n as i32,
                a.size() as i32,
                base2k as i32,
                steps as i32,
                lsh as i32,
            );
        }
    }

    fn vec_znx_lsh_inplace_backend<'s, 'r>(
        module: &Module<CudaNtt120Backend>,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        if a.size() == 0 {
            return;
        }
        let n = module.n();
        let steps = k / base2k;
        let k_rem = k % base2k;
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *mut i64;
        unsafe {
            ntt120_vec_znx_lsh_inplace(
                stream_raw,
                a_ptr,
                n as i32,
                a.size() as i32,
                base2k as i32,
                steps as i32,
                k_rem as i32,
            );
        }
    }

    // ── rotate ───────────────────────────────────────────────────────────────

    fn vec_znx_rotate_backend<'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        k: i64,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = module.n();
        let min_size = res.size().min(a.size());
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        if min_size > 0 {
            unsafe {
                ntt120_vec_znx_rotate(stream_raw, res_ptr, a_ptr, n as i32, min_size as i32, k);
            }
        }
        if res.size() > min_size {
            let zero_ptr = unsafe { res_ptr.add(min_size * n) } as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, (res.size() - min_size) * n * size_of::<i64>(), stream_raw);
            }
        }
    }

    fn vec_znx_rotate_inplace_tmp_bytes(_module: &Module<CudaNtt120Backend>) -> usize {
        0
    }

    fn vec_znx_rotate_inplace_backend<'s, 'r>(
        module: &Module<CudaNtt120Backend>,
        k: i64,
        a: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        if a.size() == 0 {
            return;
        }
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let a_col_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n);

        // Allocate temp, rotate a_col → tmp, then copy tmp → a_col.
        let byte_count = a.size() * n * size_of::<i64>();
        let tmp = CudaNtt120Backend::alloc_bytes(byte_count);
        let tmp_ptr = tmp.raw_device_ptr(0) as *mut i64;

        unsafe {
            ntt120_vec_znx_rotate(stream_raw, tmp_ptr, a_col_ptr as *const i64, n as i32, a.size() as i32, k);
            cudaMemcpyAsync(
                a_col_ptr as *mut std::ffi::c_void,
                tmp_ptr as *const std::ffi::c_void,
                byte_count,
                3, /*DeviceToDevice*/
                stream_raw,
            );
        }
        stream
            .synchronize()
            .expect("CUDA sync failed in vec_znx_rotate_inplace_backend");
        drop(tmp);
    }

    // ── automorphism ─────────────────────────────────────────────────────────

    fn vec_znx_automorphism_backend<'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        k: i64,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = module.n();
        let min_size = res.size().min(a.size());
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        if min_size > 0 {
            unsafe {
                ntt120_vec_znx_automorphism(stream_raw, res_ptr, a_ptr, n as i32, min_size as i32, k);
            }
        }
        if res.size() > min_size {
            let zero_ptr = unsafe { res_ptr.add(min_size * n) } as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, (res.size() - min_size) * n * size_of::<i64>(), stream_raw);
            }
        }
    }

    fn vec_znx_automorphism_inplace_tmp_bytes(_module: &Module<CudaNtt120Backend>) -> usize {
        0
    }

    fn vec_znx_automorphism_inplace<'s, 'r>(
        module: &Module<CudaNtt120Backend>,
        k: i64,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        if res.size() == 0 {
            return;
        }
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let col_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n);

        let byte_count = res.size() * n * size_of::<i64>();
        let tmp = CudaNtt120Backend::alloc_bytes(byte_count);
        let tmp_ptr = tmp.raw_device_ptr(0) as *mut i64;

        unsafe {
            ntt120_vec_znx_automorphism(stream_raw, tmp_ptr, col_ptr as *const i64, n as i32, res.size() as i32, k);
            cudaMemcpyAsync(
                col_ptr as *mut std::ffi::c_void,
                tmp_ptr as *const std::ffi::c_void,
                byte_count,
                3, /*DeviceToDevice*/
                stream_raw,
            );
        }
        stream
            .synchronize()
            .expect("CUDA sync failed in vec_znx_automorphism_inplace");
        drop(tmp);
    }

    // ── mul_xp_minus_one ─────────────────────────────────────────────────────

    fn vec_znx_mul_xp_minus_one_backend(
        module: &Module<CudaNtt120Backend>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = module.n();
        let min_size = res.size().min(a.size());
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n) as *const i64;
        if min_size > 0 {
            unsafe {
                ntt120_vec_znx_mul_xp_minus_one(stream_raw, res_ptr, a_ptr, n as i32, min_size as i32, k);
            }
        }
        if res.size() > min_size {
            let zero_ptr = unsafe { res_ptr.add(min_size * n) } as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, (res.size() - min_size) * n * size_of::<i64>(), stream_raw);
            }
        }
    }

    fn vec_znx_mul_xp_minus_one_inplace_tmp_bytes(_module: &Module<CudaNtt120Backend>) -> usize {
        0
    }

    fn vec_znx_mul_xp_minus_one_inplace_backend<'s>(
        module: &Module<CudaNtt120Backend>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        if res.size() == 0 {
            return;
        }
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let col_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n);

        let byte_count = res.size() * n * size_of::<i64>();
        let tmp = CudaNtt120Backend::alloc_bytes(byte_count);
        let tmp_ptr = tmp.raw_device_ptr(0) as *mut i64;

        // tmp = (X^k - 1) * res_col  (using current res as source)
        unsafe {
            ntt120_vec_znx_mul_xp_minus_one(stream_raw, tmp_ptr, col_ptr as *const i64, n as i32, res.size() as i32, k);
            cudaMemcpyAsync(
                col_ptr as *mut std::ffi::c_void,
                tmp_ptr as *const std::ffi::c_void,
                byte_count,
                3, /*DeviceToDevice*/
                stream_raw,
            );
        }
        stream
            .synchronize()
            .expect("CUDA sync failed in vec_znx_mul_xp_minus_one_inplace_backend");
        drop(tmp);
    }

    // ── ring split/merge/switch (not yet GPU-native) ─────────────────────────

    fn vec_znx_split_ring_tmp_bytes(_module: &Module<CudaNtt120Backend>) -> usize {
        0
    }

    fn vec_znx_split_ring_backend<'s>(
        module: &Module<CudaNtt120Backend>,
        res: &mut [VecZnxBackendMut<'_, CudaNtt120Backend>],
        res_col: usize,
        a: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        assert_eq!(res.len(), 2, "split_ring requires exactly 2 output buffers");
        let n_full = module.n();
        let n_half = n_full / 2;
        let nlimbs = res[0].size().min(res[1].size()).min(a.size());
        if nlimbs == 0 {
            return;
        }
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n_full) as *const i64;
        // Split res into two mutable references
        let (res_even_slice, res_odd_slice) = res.split_at_mut(1);
        let res_even = &mut res_even_slice[0];
        let res_odd = &mut res_odd_slice[0];
        let even_buf: &CudaBuf = unsafe { res_even.data.ptr.as_ref() };
        let odd_buf: &CudaBuf = unsafe { res_odd.data.ptr.as_ref() };
        let even_ptr = vec_znx_col_ptr(even_buf, res_even.data.offset, res_col, res_even.size(), n_half) as *mut i64;
        let odd_ptr = vec_znx_col_ptr(odd_buf, res_odd.data.offset, res_col, res_odd.size(), n_half) as *mut i64;
        unsafe {
            ntt120_vec_znx_split_ring(
                stream_raw,
                even_ptr,
                odd_ptr,
                a_ptr,
                n_full as i32,
                n_half as i32,
                nlimbs as i32,
            );
        }
    }

    fn vec_znx_merge_rings_tmp_bytes(_module: &Module<CudaNtt120Backend>) -> usize {
        0
    }

    fn vec_znx_merge_rings_backend<'s>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &[VecZnxBackendRef<'_, CudaNtt120Backend>],
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        assert_eq!(a.len(), 2, "merge_rings requires exactly 2 input buffers");
        let n_full = module.n();
        let n_half = n_full / 2;
        let nlimbs = res.size().min(a[0].size()).min(a[1].size());
        if nlimbs == 0 {
            return;
        }
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let even_buf: &CudaBuf = unsafe { a[0].data.ptr.as_ref() };
        let odd_buf: &CudaBuf = unsafe { a[1].data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n_full) as *mut i64;
        let even_ptr = vec_znx_col_ptr(even_buf, a[0].data.offset, a_col, a[0].size(), n_half) as *const i64;
        let odd_ptr = vec_znx_col_ptr(odd_buf, a[1].data.offset, a_col, a[1].size(), n_half) as *const i64;
        unsafe {
            ntt120_vec_znx_merge_rings(
                stream_raw,
                res_ptr,
                even_ptr,
                odd_ptr,
                n_full as i32,
                n_half as i32,
                nlimbs as i32,
            );
        }
    }

    fn vec_znx_switch_ring_backend(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n_src = module.n();
        // The res module degree may differ; infer n_dst from res buffer n
        let n_dst = res.n();
        let nlimbs = res.size().min(a.size());
        if nlimbs == 0 || n_src == n_dst {
            if n_src == n_dst && nlimbs > 0 {
                // Identity: just copy
                let stream = cuda_stream();
                let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
                let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
                let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
                let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n_src);
                let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n_src);
                unsafe {
                    cudaMemcpyAsync(res_ptr as _, a_ptr as _, nlimbs * n_src * size_of::<i64>(), 3, stream_raw);
                }
            }
            return;
        }
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n_dst) as *mut i64;
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n_src) as *const i64;
        let direction = if n_dst < n_src { 0i32 } else { 1i32 };
        unsafe {
            ntt120_vec_znx_switch_ring(
                stream_raw,
                res_ptr,
                a_ptr,
                n_src as i32,
                n_dst as i32,
                nlimbs as i32,
                direction,
            );
        }
    }

    // ── copy ─────────────────────────────────────────────────────────────────

    fn vec_znx_copy_backend(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = module.n();
        let min_size = res.size().min(a.size());
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, res.size(), n);
        let a_ptr = vec_znx_col_ptr(a_buf, a.data.offset, a_col, a.size(), n);
        if min_size > 0 {
            let copy_bytes = min_size * n * size_of::<i64>();
            unsafe {
                cudaMemcpyAsync(
                    res_ptr as *mut std::ffi::c_void,
                    a_ptr as *const std::ffi::c_void,
                    copy_bytes,
                    3, /*DeviceToDevice*/
                    stream_raw,
                );
            }
        }
        if res.size() > min_size {
            let zero_ptr = (res_ptr as usize + min_size * n * size_of::<i64>()) as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, (res.size() - min_size) * n * size_of::<i64>(), stream_raw);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_copy_range_backend(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        res_limb: usize,
        res_offset: usize,
        a: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
        a_limb: usize,
        a_offset: usize,
        len: usize,
    ) {
        if len == 0 {
            return;
        }
        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_start =
            res.data.offset + (res_col * res.size() + res_limb) * n * size_of::<i64>() + res_offset * size_of::<i64>();
        let a_start = a.data.offset + (a_col * a.size() + a_limb) * n * size_of::<i64>() + a_offset * size_of::<i64>();
        let copy_bytes = len * size_of::<i64>();
        unsafe {
            cudaMemcpyAsync(
                buf_device_ptr(res_buf, res_start) as *mut std::ffi::c_void,
                buf_device_ptr(a_buf, a_start) as *const std::ffi::c_void,
                copy_bytes,
                3, /*DeviceToDevice*/
                stream_raw,
            );
        }
    }

    // ── vec_znx sampling — CPU generation + H2D upload ───────────────────────

    fn vec_znx_fill_uniform_backend(
        module: &Module<CudaNtt120Backend>,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        seed: [u8; 32],
    ) {
        let n = module.n();
        let size = res.size();
        let mut source = Source::new(seed);
        let mut buf: Vec<i64> = vec![0i64; size * n];
        for j in 0..size {
            znx_fill_uniform_ref(base2k, &mut buf[j * n..(j + 1) * n], &mut source);
        }
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let dev_ptr = vec_znx_col_ptr(res_buf, res.data.offset, res_col, size, n) as *mut std::ffi::c_void;
        unsafe {
            cudaMemcpy(dev_ptr, buf.as_ptr() as _, size * n * size_of::<i64>(), 1 /*H2D*/);
        }
    }

    fn vec_znx_fill_normal_backend(
        module: &Module<CudaNtt120Backend>,
        res_base2k: usize,
        res: &mut VecZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) {
        let n = module.n();
        let size = res.size();
        let mut source = Source::new(seed);
        let (limb, scale) = noise_infos.target_limb_and_scale(res_base2k);
        let mut buf: Vec<i64> = vec![0i64; n];
        znx_fill_normal_f64_ref(&mut buf, noise_infos.sigma * scale, noise_infos.bound * scale, &mut source);
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let limb_offset = res.data.offset + (res_col * size + limb) * n * size_of::<i64>();
        let dev_ptr = buf_device_ptr(res_buf, limb_offset) as *mut std::ffi::c_void;
        unsafe {
            cudaMemcpy(dev_ptr, buf.as_ptr() as _, n * size_of::<i64>(), 1 /*H2D*/);
        }
    }

    fn vec_znx_add_normal_backend(
        module: &Module<CudaNtt120Backend>,
        res_base2k: usize,
        res: &mut VecZnxBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) {
        let n = module.n();
        let size = res.size();
        let stream = cuda_stream();
        let mut source = Source::new(seed);
        let (limb, scale) = noise_infos.target_limb_and_scale(res_base2k);
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let limb_offset = res.data.offset + (res_col * size + limb) * n * size_of::<i64>();
        let dev_ptr = buf_device_ptr(res_buf, limb_offset) as *mut std::ffi::c_void;
        let byte_count = n * size_of::<i64>();
        // Ensure prior GPU kernels writing to this limb have completed.
        stream.synchronize().expect("CUDA sync failed in vec_znx_add_normal_backend");
        let mut buf: Vec<i64> = vec![0i64; n];
        unsafe {
            cudaMemcpy(buf.as_mut_ptr() as _, dev_ptr, byte_count, 2 /*D2H*/);
        }
        znx_add_normal_f64_ref(&mut buf, noise_infos.sigma * scale, noise_infos.bound * scale, &mut source);
        unsafe {
            cudaMemcpy(dev_ptr, buf.as_ptr() as _, byte_count, 1 /*H2D*/);
        }
    }
}
