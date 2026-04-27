//! HalVecZnxBigImpl for CudaNtt120Backend.

use std::mem::size_of;

use poulpy_cpu_ref::reference::znx::znx_fill_normal_f64_ref;
use poulpy_hal::{
    layouts::{Backend, Module, NoiseInfos, ScratchArena, VecZnxBackendMut, VecZnxBackendRef},
    oep::HalVecZnxBigImpl,
    source::Source,
};

use crate::ntt120::CudaNtt120Backend;
use crate::ntt120::hal_impl::buf_device_ptr;
use crate::ntt120::types::Big32;
use crate::{CudaBuf, cuda_stream};

// ── extern "C" kernel launchers ───────────────────────────────────────────────

unsafe extern "C" {
    fn ntt120_big_normalize(
        stream: *mut std::ffi::c_void,
        dst: *mut i64,
        src: *const u32,
        n: i32,
        a_size: i32,
        res_size: i32,
        base2k: i32,
        res_offset: i64,
    );

    fn ntt120_big_from_small(stream: *mut std::ffi::c_void, res: *mut u32, a: *const i64, len: i32);
    fn ntt120_big_add_into(stream: *mut std::ffi::c_void, res: *mut u32, a: *const u32, b: *const u32, len: i32);
    fn ntt120_big_add_assign(stream: *mut std::ffi::c_void, res: *mut u32, a: *const u32, len: i32);
    fn ntt120_big_add_small_into(stream: *mut std::ffi::c_void, res: *mut u32, a: *const u32, b: *const i64, len: i32);
    fn ntt120_big_add_small_assign(stream: *mut std::ffi::c_void, res: *mut u32, a: *const i64, len: i32);
    fn ntt120_big_sub_into(stream: *mut std::ffi::c_void, res: *mut u32, a: *const u32, b: *const u32, len: i32);
    fn ntt120_big_sub_inplace(stream: *mut std::ffi::c_void, res: *mut u32, a: *const u32, len: i32);
    fn ntt120_big_sub_negate_inplace(stream: *mut std::ffi::c_void, res: *mut u32, a: *const u32, len: i32);
    fn ntt120_big_sub_small_a(stream: *mut std::ffi::c_void, res: *mut u32, a: *const i64, b: *const u32, len: i32);
    fn ntt120_big_sub_small_b(stream: *mut std::ffi::c_void, res: *mut u32, a: *const u32, b: *const i64, len: i32);
    fn ntt120_big_sub_small_inplace(stream: *mut std::ffi::c_void, res: *mut u32, a: *const i64, len: i32);
    fn ntt120_big_sub_small_negate_inplace(stream: *mut std::ffi::c_void, res: *mut u32, a: *const i64, len: i32);
    fn ntt120_big_negate_into(stream: *mut std::ffi::c_void, res: *mut u32, a: *const u32, len: i32);
    fn ntt120_big_negate_inplace(stream: *mut std::ffi::c_void, res: *mut u32, len: i32);
    fn ntt120_big_automorphism(stream: *mut std::ffi::c_void, res: *mut u32, a: *const u32, n: i32, nlimbs: i32, p: i64);

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

// ── Layout helpers ────────────────────────────────────────────────────────────

/// Byte offset to the start of column `col` in a VecZnxBig buffer.
/// Layout: [cols × size × n] Big32
#[inline]
fn big_col_byte_offset(offset: usize, col: usize, size: usize, n: usize) -> usize {
    offset + col * size * n * size_of::<Big32>()
}

/// Byte offset to the start of limb `limb` within column `col`.
#[inline]
fn big_limb_byte_offset(offset: usize, col: usize, size: usize, n: usize, limb: usize) -> usize {
    offset + (col * size + limb) * n * size_of::<Big32>()
}

/// Device pointer to column `col`, cast to `*mut u32` (Big32 element = 4 u32s).
#[inline]
fn big_col_ptr_mut(buf: &CudaBuf, offset: usize, col: usize, size: usize, n: usize) -> *mut u32 {
    buf_device_ptr(buf, big_col_byte_offset(offset, col, size, n)) as *mut u32
}

#[inline]
fn big_col_ptr(buf: &CudaBuf, offset: usize, col: usize, size: usize, n: usize) -> *const u32 {
    buf_device_ptr(buf, big_col_byte_offset(offset, col, size, n)) as *const u32
}

#[inline]
fn big_limb_ptr(buf: &CudaBuf, offset: usize, col: usize, size: usize, n: usize, limb: usize) -> *const u32 {
    buf_device_ptr(buf, big_limb_byte_offset(offset, col, size, n, limb)) as *const u32
}

#[inline]
fn big_limb_ptr_mut(buf: &CudaBuf, offset: usize, col: usize, size: usize, n: usize, limb: usize) -> *mut u32 {
    buf_device_ptr(buf, big_limb_byte_offset(offset, col, size, n, limb)) as *mut u32
}

/// VecZnx column pointer as i64 (for from_small/add_small_* kernels).
#[inline]
fn vec_znx_limb_ptr(buf: &CudaBuf, offset: usize, col: usize, size: usize, n: usize, limb: usize) -> *const i64 {
    buf_device_ptr(buf, offset + (col * size + limb) * n * size_of::<i64>()) as *const i64
}

// ── HalVecZnxBigImpl ─────────────────────────────────────────────────────────

unsafe impl HalVecZnxBigImpl<CudaNtt120Backend> for CudaNtt120Backend {
    fn vec_znx_big_from_small_backend(
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = res.n();
        let min_size = res.size().min(a.size());
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };

        if min_size > 0 {
            let res_ptr = big_col_ptr_mut(res_buf, res.data.offset, res_col, res.size(), n);
            let a_ptr = vec_znx_limb_ptr(a_buf, a.data.offset, a_col, a.size(), n, 0);
            unsafe {
                ntt120_big_from_small(stream_raw, res_ptr, a_ptr, (min_size * n) as i32);
            }
        }
        // Zero extra res limbs
        if res.size() > min_size {
            let zero_offset = big_limb_byte_offset(res.data.offset, res_col, res.size(), n, min_size);
            let zero_ptr = buf_device_ptr(res_buf, zero_offset) as *mut std::ffi::c_void;
            unsafe {
                cudaMemsetAsync(zero_ptr, 0, (res.size() - min_size) * n * size_of::<Big32>(), stream_raw);
            }
        }
    }

    fn vec_znx_big_add_normal_backend(
        module: &Module<CudaNtt120Backend>,
        res_base2k: usize,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) {
        let n = module.n();
        let size = res.size();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let mut source = Source::new(seed);
        let (limb, scale) = noise_infos.target_limb_and_scale(res_base2k);
        // Generate i64 noise on CPU and upload to a temporary device buffer.
        let mut noise: Vec<i64> = vec![0i64; n];
        znx_fill_normal_f64_ref(&mut noise, noise_infos.sigma * scale, noise_infos.bound * scale, &mut source);
        let noise_bytes = unsafe { std::slice::from_raw_parts(noise.as_ptr() as *const u8, n * size_of::<i64>()) };
        let tmp: CudaBuf = CudaNtt120Backend::from_host_bytes(noise_bytes);
        let tmp_ptr = tmp.raw_device_ptr(0) as *const i64;
        // GPU: Big32[i] += (i128)(i64)noise[i]
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let big_ptr = buf_device_ptr(res_buf, res.data.offset + (res_col * size + limb) * n * size_of::<Big32>()) as *mut u32;
        unsafe {
            ntt120_big_add_small_assign(stream_raw, big_ptr, tmp_ptr, n as i32);
        }
        stream
            .synchronize()
            .expect("CUDA sync failed in vec_znx_big_add_normal_backend");
        drop(tmp);
    }

    fn vec_znx_big_add_into(
        _module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
        b: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, CudaNtt120Backend>,
        b_col: usize,
    ) {
        let n = res.n();
        let res_size = res.size();
        let a_size = a.size();
        let b_size = b.size();
        let sum_size = res_size.min(a_size).min(b_size);
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let b_buf: &CudaBuf = unsafe { b.data.ptr.as_ref() };

        // Add overlapping limbs
        if sum_size > 0 {
            let res_ptr = big_col_ptr_mut(res_buf, res.data.offset, res_col, res_size, n);
            let a_ptr = big_col_ptr(a_buf, a.data.offset, a_col, a_size, n);
            let b_ptr = big_col_ptr(b_buf, b.data.offset, b_col, b_size, n);
            unsafe {
                ntt120_big_add_into(stream_raw, res_ptr, a_ptr, b_ptr, (sum_size * n) as i32);
            }
        }

        // Copy extra limbs from whichever is larger
        if a_size <= b_size {
            let extra_start = sum_size;
            let extra_end = b_size.min(res_size);
            if extra_end > extra_start {
                let len = (extra_end - extra_start) * n * size_of::<Big32>();
                let res_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, extra_start);
                let b_ptr = big_limb_ptr(b_buf, b.data.offset, b_col, b_size, n, extra_start);
                unsafe {
                    cudaMemcpyAsync(res_ptr as _, b_ptr as _, len, 3, stream_raw);
                }
            }
            let zero_start = extra_end;
            if res_size > zero_start {
                let zero_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, zero_start);
                unsafe {
                    cudaMemsetAsync(zero_ptr as _, 0, (res_size - zero_start) * n * size_of::<Big32>(), stream_raw);
                }
            }
        } else {
            let extra_start = sum_size;
            let extra_end = a_size.min(res_size);
            if extra_end > extra_start {
                let len = (extra_end - extra_start) * n * size_of::<Big32>();
                let res_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, extra_start);
                let a_ptr = big_limb_ptr(a_buf, a.data.offset, a_col, a_size, n, extra_start);
                unsafe {
                    cudaMemcpyAsync(res_ptr as _, a_ptr as _, len, 3, stream_raw);
                }
            }
            let zero_start = extra_end;
            if res_size > zero_start {
                let zero_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, zero_start);
                unsafe {
                    cudaMemsetAsync(zero_ptr as _, 0, (res_size - zero_start) * n * size_of::<Big32>(), stream_raw);
                }
            }
        }
    }

    fn vec_znx_big_add_assign(
        _module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = res.n();
        let min_size = res.size().min(a.size());
        if min_size == 0 {
            return;
        }
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = big_col_ptr_mut(res_buf, res.data.offset, res_col, res.size(), n);
        let a_ptr = big_col_ptr(a_buf, a.data.offset, a_col, a.size(), n);
        unsafe {
            ntt120_big_add_assign(stream_raw, res_ptr, a_ptr, (min_size * n) as i32);
        }
    }

    fn vec_znx_big_add_small_into_backend(
        _module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        b_col: usize,
    ) {
        let n = res.n();
        let res_size = res.size();
        let a_size = a.size();
        let b_size = b.size();
        let sum_size = res_size.min(a_size).min(b_size);
        let a_cpy = a_size.min(res_size);
        let b_cpy = b_size.min(res_size);
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let b_buf: &CudaBuf = unsafe { b.data.ptr.as_ref() };

        // a + b (i128 + i64)
        if sum_size > 0 {
            let res_ptr = big_col_ptr_mut(res_buf, res.data.offset, res_col, res_size, n);
            let a_ptr = big_col_ptr(a_buf, a.data.offset, a_col, a_size, n);
            let b_ptr = vec_znx_limb_ptr(b_buf, b.data.offset, b_col, b_size, n, 0);
            unsafe {
                ntt120_big_add_small_into(stream_raw, res_ptr, a_ptr, b_ptr, (sum_size * n) as i32);
            }
        }
        // copy extra a limbs [sum_size..a_cpy)
        if a_cpy > sum_size {
            let res_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, sum_size);
            let a_ptr = big_limb_ptr(a_buf, a.data.offset, a_col, a_size, n, sum_size);
            unsafe {
                cudaMemcpyAsync(
                    res_ptr as _,
                    a_ptr as _,
                    (a_cpy - sum_size) * n * size_of::<Big32>(),
                    3,
                    stream_raw,
                );
            }
        }
        // sign-extend b limbs [a_cpy..b_cpy) into Big32
        if b_cpy > a_cpy {
            let res_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, a_cpy);
            let b_ptr = vec_znx_limb_ptr(b_buf, b.data.offset, b_col, b_size, n, a_cpy);
            unsafe {
                ntt120_big_from_small(stream_raw, res_ptr, b_ptr, ((b_cpy - a_cpy) * n) as i32);
            }
        }
        // zero remaining
        let zero_start = a_cpy.max(b_cpy);
        if res_size > zero_start {
            let zero_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, zero_start);
            unsafe {
                cudaMemsetAsync(zero_ptr as _, 0, (res_size - zero_start) * n * size_of::<Big32>(), stream_raw);
            }
        }
    }

    fn vec_znx_big_add_small_assign<'r, 'a>(
        _module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = res.n();
        let min_size = res.size().min(a.size());
        if min_size == 0 {
            return;
        }
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = big_col_ptr_mut(res_buf, res.data.offset, res_col, res.size(), n);
        let a_ptr = vec_znx_limb_ptr(a_buf, a.data.offset, a_col, a.size(), n, 0);
        unsafe {
            ntt120_big_add_small_assign(stream_raw, res_ptr, a_ptr, (min_size * n) as i32);
        }
    }

    fn vec_znx_big_sub(
        _module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
        b: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, CudaNtt120Backend>,
        b_col: usize,
    ) {
        let n = res.n();
        let res_size = res.size();
        let a_size = a.size();
        let b_size = b.size();
        let sum_size = res_size.min(a_size).min(b_size);
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let b_buf: &CudaBuf = unsafe { b.data.ptr.as_ref() };

        if sum_size > 0 {
            let res_ptr = big_col_ptr_mut(res_buf, res.data.offset, res_col, res_size, n);
            let a_ptr = big_col_ptr(a_buf, a.data.offset, a_col, a_size, n);
            let b_ptr = big_col_ptr(b_buf, b.data.offset, b_col, b_size, n);
            unsafe {
                ntt120_big_sub_into(stream_raw, res_ptr, a_ptr, b_ptr, (sum_size * n) as i32);
            }
        }

        if a_size >= b_size {
            let extra_end = a_size.min(res_size);
            if extra_end > sum_size {
                let res_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, sum_size);
                let a_ptr = big_limb_ptr(a_buf, a.data.offset, a_col, a_size, n, sum_size);
                unsafe {
                    cudaMemcpyAsync(
                        res_ptr as _,
                        a_ptr as _,
                        (extra_end - sum_size) * n * size_of::<Big32>(),
                        3,
                        stream_raw,
                    );
                }
            }
            if res_size > extra_end {
                let zero_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, extra_end);
                unsafe {
                    cudaMemsetAsync(zero_ptr as _, 0, (res_size - extra_end) * n * size_of::<Big32>(), stream_raw);
                }
            }
        } else {
            // a < b: extra b limbs get negated
            let extra_end = b_size.min(res_size);
            if extra_end > sum_size {
                let res_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, sum_size);
                let b_ptr = big_limb_ptr(b_buf, b.data.offset, b_col, b_size, n, sum_size);
                unsafe {
                    ntt120_big_negate_into(stream_raw, res_ptr, b_ptr, ((extra_end - sum_size) * n) as i32);
                }
            }
            if res_size > extra_end {
                let zero_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, extra_end);
                unsafe {
                    cudaMemsetAsync(zero_ptr as _, 0, (res_size - extra_end) * n * size_of::<Big32>(), stream_raw);
                }
            }
        }
    }

    fn vec_znx_big_sub_inplace(
        _module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = res.n();
        let min_size = res.size().min(a.size());
        if min_size == 0 {
            return;
        }
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = big_col_ptr_mut(res_buf, res.data.offset, res_col, res.size(), n);
        let a_ptr = big_col_ptr(a_buf, a.data.offset, a_col, a.size(), n);
        unsafe {
            ntt120_big_sub_inplace(stream_raw, res_ptr, a_ptr, (min_size * n) as i32);
        }
    }

    fn vec_znx_big_sub_negate_inplace(
        _module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = res.n();
        let res_size = res.size();
        let a_size = a.size();
        let sum_size = res_size.min(a_size);
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };

        if sum_size > 0 {
            let res_ptr = big_col_ptr_mut(res_buf, res.data.offset, res_col, res_size, n);
            let a_ptr = big_col_ptr(a_buf, a.data.offset, a_col, a_size, n);
            unsafe {
                ntt120_big_sub_negate_inplace(stream_raw, res_ptr, a_ptr, (sum_size * n) as i32);
            }
        }
        // Extra res limbs (beyond a): negate
        if res_size > a_size {
            let extra_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, a_size);
            unsafe {
                ntt120_big_negate_inplace(stream_raw, extra_ptr, ((res_size - a_size) * n) as i32);
            }
        }
    }

    fn vec_znx_big_sub_small_a_backend(
        _module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
        b: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, CudaNtt120Backend>,
        b_col: usize,
    ) {
        let n = res.n();
        let res_size = res.size();
        let a_size = a.size();
        let b_size = b.size();
        let sum_size = res_size.min(a_size).min(b_size);
        let a_cpy = a_size.min(res_size);
        let b_cpy = b_size.min(res_size);
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let b_buf: &CudaBuf = unsafe { b.data.ptr.as_ref() };

        // (i64)a - (i128)b in overlap
        if sum_size > 0 {
            let res_ptr = big_col_ptr_mut(res_buf, res.data.offset, res_col, res_size, n);
            let a_ptr = vec_znx_limb_ptr(a_buf, a.data.offset, a_col, a_size, n, 0);
            let b_ptr = big_col_ptr(b_buf, b.data.offset, b_col, b_size, n);
            unsafe {
                ntt120_big_sub_small_a(stream_raw, res_ptr, a_ptr, b_ptr, (sum_size * n) as i32);
            }
        }
        // sign-extend extra a limbs [sum_size..a_cpy)
        if a_cpy > sum_size {
            let res_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, sum_size);
            let a_ptr = vec_znx_limb_ptr(a_buf, a.data.offset, a_col, a_size, n, sum_size);
            unsafe {
                ntt120_big_from_small(stream_raw, res_ptr, a_ptr, ((a_cpy - sum_size) * n) as i32);
            }
        }
        // negate extra b limbs [sum_size..b_cpy) if beyond a_cpy
        if b_cpy > sum_size {
            let extra_start = sum_size.max(a_cpy);
            if b_cpy > extra_start {
                let res_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, extra_start);
                let b_ptr = big_limb_ptr(b_buf, b.data.offset, b_col, b_size, n, extra_start);
                unsafe {
                    ntt120_big_negate_into(stream_raw, res_ptr, b_ptr, ((b_cpy - extra_start) * n) as i32);
                }
            }
        }
        let zero_start = a_cpy.max(b_cpy);
        if res_size > zero_start {
            let zero_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, zero_start);
            unsafe {
                cudaMemsetAsync(zero_ptr as _, 0, (res_size - zero_start) * n * size_of::<Big32>(), stream_raw);
            }
        }
    }

    fn vec_znx_big_sub_small_inplace<'r, 'a>(
        _module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = res.n();
        let min_size = res.size().min(a.size());
        if min_size == 0 {
            return;
        }
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let res_ptr = big_col_ptr_mut(res_buf, res.data.offset, res_col, res.size(), n);
        let a_ptr = vec_znx_limb_ptr(a_buf, a.data.offset, a_col, a.size(), n, 0);
        unsafe {
            ntt120_big_sub_small_inplace(stream_raw, res_ptr, a_ptr, (min_size * n) as i32);
        }
    }

    fn vec_znx_big_sub_small_b_backend(
        _module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, CudaNtt120Backend>,
        b_col: usize,
    ) {
        let n = res.n();
        let res_size = res.size();
        let a_size = a.size();
        let b_size = b.size();
        let sum_size = res_size.min(a_size).min(b_size);
        let a_cpy = a_size.min(res_size);
        let b_cpy = b_size.min(res_size);
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let b_buf: &CudaBuf = unsafe { b.data.ptr.as_ref() };

        // (i128)a - (i64)b in overlap
        if sum_size > 0 {
            let res_ptr = big_col_ptr_mut(res_buf, res.data.offset, res_col, res_size, n);
            let a_ptr = big_col_ptr(a_buf, a.data.offset, a_col, a_size, n);
            let b_ptr = vec_znx_limb_ptr(b_buf, b.data.offset, b_col, b_size, n, 0);
            unsafe {
                ntt120_big_sub_small_b(stream_raw, res_ptr, a_ptr, b_ptr, (sum_size * n) as i32);
            }
        }
        // copy extra a limbs
        if a_cpy > sum_size {
            let res_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, sum_size);
            let a_ptr = big_limb_ptr(a_buf, a.data.offset, a_col, a_size, n, sum_size);
            unsafe {
                cudaMemcpyAsync(
                    res_ptr as _,
                    a_ptr as _,
                    (a_cpy - sum_size) * n * size_of::<Big32>(),
                    3,
                    stream_raw,
                );
            }
        }
        // negate-from-small extra b limbs (-(i64) as i128) [a_cpy..b_cpy)
        if b_cpy > a_cpy {
            let res_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, a_cpy);
            let b_ptr = vec_znx_limb_ptr(b_buf, b.data.offset, b_col, b_size, n, a_cpy);
            // res = -b (sign-extend then negate): use from_small then negate_inplace
            unsafe {
                ntt120_big_from_small(stream_raw, res_ptr, b_ptr, ((b_cpy - a_cpy) * n) as i32);
                ntt120_big_negate_inplace(stream_raw, res_ptr, ((b_cpy - a_cpy) * n) as i32);
            }
        }
        let zero_start = a_cpy.max(b_cpy);
        if res_size > zero_start {
            let zero_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, zero_start);
            unsafe {
                cudaMemsetAsync(zero_ptr as _, 0, (res_size - zero_start) * n * size_of::<Big32>(), stream_raw);
            }
        }
    }

    fn vec_znx_big_sub_small_negate_inplace<'r, 'a>(
        _module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'r, CudaNtt120Backend>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = res.n();
        let min_size = res.size().min(a.size());
        let res_size = res.size();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };

        if min_size > 0 {
            let res_ptr = big_col_ptr_mut(res_buf, res.data.offset, res_col, res_size, n);
            let a_ptr = vec_znx_limb_ptr(a_buf, a.data.offset, a_col, a.size(), n, 0);
            unsafe {
                ntt120_big_sub_small_negate_inplace(stream_raw, res_ptr, a_ptr, (min_size * n) as i32);
            }
        }
        // Extra res limbs (beyond a): negate
        if res_size > min_size {
            let extra_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res_size, n, min_size);
            unsafe {
                ntt120_big_negate_inplace(stream_raw, extra_ptr, ((res_size - min_size) * n) as i32);
            }
        }
    }

    fn vec_znx_big_negate(
        _module: &Module<CudaNtt120Backend>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = res.n();
        let min_size = res.size().min(a.size());
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };

        if min_size > 0 {
            let res_ptr = big_col_ptr_mut(res_buf, res.data.offset, res_col, res.size(), n);
            let a_ptr = big_col_ptr(a_buf, a.data.offset, a_col, a.size(), n);
            unsafe {
                ntt120_big_negate_into(stream_raw, res_ptr, a_ptr, (min_size * n) as i32);
            }
        }
        if res.size() > min_size {
            let zero_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res.size(), n, min_size);
            unsafe {
                cudaMemsetAsync(zero_ptr as _, 0, (res.size() - min_size) * n * size_of::<Big32>(), stream_raw);
            }
        }
    }

    fn vec_znx_big_negate_inplace(
        _module: &Module<CudaNtt120Backend>,
        a: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = a.n();
        if a.size() == 0 {
            return;
        }
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let a_ptr = big_col_ptr_mut(a_buf, a.data.offset, a_col, a.size(), n);
        unsafe {
            ntt120_big_negate_inplace(stream_raw, a_ptr, (a.size() * n) as i32);
        }
    }

    fn vec_znx_big_normalize_tmp_bytes(_module: &Module<CudaNtt120Backend>) -> usize {
        0
    }

    fn vec_znx_big_normalize<'s, 'r, 'a>(
        module: &Module<CudaNtt120Backend>,
        res: &mut VecZnxBackendMut<'r, CudaNtt120Backend>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxBigBackendRef<'a, CudaNtt120Backend>,
        a_base2k: usize,
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        assert_eq!(
            res_base2k, a_base2k,
            "CudaNtt120Backend::vec_znx_big_normalize: cross-base2k not implemented"
        );

        let n = module.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

        let a_byte_offset = a.data.offset + (a_col * a.size()) * n * size_of::<Big32>();
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let a_ptr = buf_device_ptr(a_buf, a_byte_offset) as *const u32;

        let res_byte_offset = res.data.offset + (res_col * res.size()) * n * size_of::<i64>();
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let res_ptr = buf_device_ptr(res_buf, res_byte_offset) as *mut i64;

        unsafe {
            ntt120_big_normalize(
                stream_raw,
                res_ptr,
                a_ptr,
                n as i32,
                a.size() as i32,
                res.size() as i32,
                res_base2k as i32,
                res_offset,
            );
        }
    }

    fn vec_znx_big_automorphism(
        _module: &Module<CudaNtt120Backend>,
        k: i64,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, CudaNtt120Backend>,
        a_col: usize,
    ) {
        let n = res.n();
        let min_size = res.size().min(a.size());
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let res_buf: &CudaBuf = unsafe { res.data.ptr.as_ref() };
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };

        if min_size > 0 {
            let res_ptr = big_col_ptr_mut(res_buf, res.data.offset, res_col, res.size(), n);
            let a_ptr = big_col_ptr(a_buf, a.data.offset, a_col, a.size(), n);
            unsafe {
                ntt120_big_automorphism(stream_raw, res_ptr, a_ptr, n as i32, min_size as i32, k);
            }
        }
        if res.size() > min_size {
            let zero_ptr = big_limb_ptr_mut(res_buf, res.data.offset, res_col, res.size(), n, min_size);
            unsafe {
                cudaMemsetAsync(zero_ptr as _, 0, (res.size() - min_size) * n * size_of::<Big32>(), stream_raw);
            }
        }
    }

    fn vec_znx_big_automorphism_inplace_tmp_bytes(_module: &Module<CudaNtt120Backend>) -> usize {
        0
    }

    fn vec_znx_big_automorphism_inplace<'s>(
        _module: &Module<CudaNtt120Backend>,
        k: i64,
        a: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, CudaNtt120Backend>,
        a_col: usize,
        _scratch: &mut ScratchArena<'s, CudaNtt120Backend>,
    ) {
        if a.size() == 0 {
            return;
        }
        let n = a.n();
        let stream = cuda_stream();
        let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
        let a_buf: &CudaBuf = unsafe { a.data.ptr.as_ref() };
        let col_ptr = big_col_ptr_mut(a_buf, a.data.offset, a_col, a.size(), n);

        let byte_count = a.size() * n * size_of::<Big32>();
        let tmp = CudaNtt120Backend::alloc_bytes(byte_count);
        let tmp_ptr = tmp.raw_device_ptr(0) as *mut u32;

        unsafe {
            ntt120_big_automorphism(stream_raw, tmp_ptr, col_ptr as *const u32, n as i32, a.size() as i32, k);
            cudaMemcpyAsync(col_ptr as _, tmp_ptr as _, byte_count, 3, stream_raw);
        }
        stream
            .synchronize()
            .expect("CUDA sync failed in vec_znx_big_automorphism_inplace");
        drop(tmp);
    }
}
