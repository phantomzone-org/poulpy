//! NTT120 CUDA backend — backend type, buffer wiring, and `bytes_of_*` overrides.
//!
//! `CudaNtt120Backend` is separate from `CudaGpuBackend` (FFT64) because the
//! two backends use different prepared-domain layouts: `ScalarPrep = i32`
//! (Montgomery residues mod four ~30-bit primes) vs `f64` (FFT coefficients).
//! All per-family `bytes_of_*` methods are overridden to match the NTT120
//! device layouts documented in `doc/ntt120-gpu-design.md`.

pub mod hal_big_impl;
pub mod hal_cnv_impl;
pub mod hal_impl;
pub mod hal_svp_impl;
pub mod hal_vec_znx_impl;
pub mod hal_vmp_impl;
pub mod twiddle;
pub mod types;

use std::{mem::size_of, ptr::NonNull};

use cudarc::driver::CudaSlice;

use crate::{CudaBuf, CudaBufMut, CudaBufRef, cuda_stream};
use poulpy_hal::{layouts::{Backend, Device, Module}, oep::HalModuleImpl};
use types::Big32;

/// NTT120 CUDA backend marker type.
///
/// Uses four ~30-bit CRT primes (Primes30, Q ≈ 2^120) for polynomial
/// arithmetic on device. Buffer management is shared with
/// [`CudaGpuBackend`](crate::CudaGpuBackend); only the prepared-domain
/// scalar types and layout sizes differ.
#[derive(Debug, Clone, Copy)]
pub struct CudaNtt120Backend;

/// Handle owned by `Module<CudaNtt120Backend>`.
///
/// Allocated once at module creation. All `CudaSlice` fields live on device
/// and are never modified after construction.
pub struct CudaNtt120Handle {
    /// Forward twiddle factors: prime-major `[4 × n]` Montgomery `u32`.
    pub(crate) twiddle_fwd: CudaSlice<u32>,
    /// Forward twiddle MSB: `[4 × (n / LSB_SIZE)]` — one per shared-memory block.
    /// `None` when `n < LSB_SIZE` (only possible for tiny test dimensions).
    pub(crate) twiddle_fwd_msb: Option<CudaSlice<u32>>,
    /// Inverse twiddle factors: prime-major `[4 × n]` Montgomery `u32`.
    pub(crate) twiddle_inv: CudaSlice<u32>,
    /// Inverse twiddle MSB: `[4 × (n / LSB_SIZE)]`.
    pub(crate) twiddle_inv_msb: Option<CudaSlice<u32>>,
    /// `n^{-1} mod Q[k]` in Montgomery form, one per prime — device copy for INTTPhase2.
    pub(crate) inv_n_mont: CudaSlice<u32>,
    /// Primes30 moduli `Q[k]` — device copy for `__ldg` loads in kernels.
    pub(crate) primes: CudaSlice<u32>,
    /// `Q[k]^{-1} mod 2^32` as `i32` — device copy for `__ldg` loads in kernels.
    pub(crate) inv_primes: CudaSlice<i32>,
    pub(crate) log_n: usize,
}

impl Backend for CudaNtt120Backend {
    /// Montgomery residues mod four ~30-bit primes stored as `i32`.
    type ScalarPrep = i32;
    /// 128-bit CRT accumulator stored as four `u32` words.
    type ScalarBig = Big32;

    // Buffer types are shared with `CudaGpuBackend` — both backends store
    // opaque bytes and rely on kernel-level conventions for layout.
    type OwnedBuf = CudaBuf;
    type BufRef<'a> = CudaBufRef<'a>;
    type BufMut<'a> = CudaBufMut<'a>;

    type Handle = CudaNtt120Handle;
    type Location = Device;

    /// CUDA L1/shared memory aligns to 128 bytes.
    const SCRATCH_ALIGN: usize = 128;

    // -----------------------------------------------------------------------
    // bytes_of_* overrides
    //
    // Every prepared-domain family has a layout that differs from the default
    // `n * cols * size * size_of::<ScalarPrep>()` formula. See the per-family
    // layout sections in `doc/ntt120-gpu-design.md`.
    // -----------------------------------------------------------------------

    /// `VecZnxDft`: `[batch][prime][coeff] : i32`, 4 primes → ×4 the default.
    fn bytes_of_vec_znx_dft(n: usize, cols: usize, size: usize) -> usize {
        4 * n * cols * size * size_of::<i32>()
    }

    // `VecZnxBig`: `[batch][coeff][word] : u32`.
    // `size_of::<Big32>() == 16`, so the default formula gives the correct
    // result with no override needed.

    /// `SvpPPol`: `[col][coeff][prime][lane] : u32`, 4 primes × 2 Montgomery lanes.
    fn bytes_of_svp_ppol(n: usize, cols: usize) -> usize {
        n * cols * 4 * 2 * size_of::<u32>()
    }

    /// `VmpPMat`: `[blk][out_vec][in_row][q120c_lane] : u32`.
    /// blk = n/2 coefficient pairs, lane = 8 (4 primes × 2 Montgomery lanes).
    fn bytes_of_vmp_pmat(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        let out_vecs = cols_out * size;
        let in_rows = rows * cols_in;
        (n / 2) * out_vecs * in_rows * 8 * size_of::<u32>()
    }

    /// `CnvPVecL`: `[batch][blk][lane8] : i32`, blk = n/2, lane8 = 8.
    fn bytes_of_cnv_pvec_left(n: usize, cols: usize, size: usize) -> usize {
        (n / 2) * cols * size * 8 * size_of::<i32>()
    }

    /// `CnvPVecR`: `[batch][blk][lane16] : u32`, blk = n/2, lane16 = 16.
    fn bytes_of_cnv_pvec_right(n: usize, cols: usize, size: usize) -> usize {
        (n / 2) * cols * size * 16 * size_of::<u32>()
    }

    // -----------------------------------------------------------------------
    // Buffer operations — identical to CudaGpuBackend (same CudaBuf storage).
    // -----------------------------------------------------------------------

    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        CudaBuf::alloc_device(len)
    }

    fn alloc_zeroed_bytes(len: usize) -> Self::OwnedBuf {
        CudaBuf::alloc_zeroed_device(len)
    }

    fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf {
        CudaBuf::from_host_bytes(bytes)
    }

    fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf {
        CudaBuf::from_host_bytes(&bytes)
    }

    fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8> {
        buf.to_host_vec()
    }

    fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]) {
        assert_eq!(dst.len(), buf.len);
        if !dst.is_empty() {
            let device = buf.device.lock().unwrap();
            cuda_stream()
                .memcpy_dtoh(device.as_ref().expect("missing CUDA device buffer"), dst)
                .expect("failed to copy CUDA device buffer to host");
        }
    }

    fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]) {
        assert_eq!(src.len(), buf.len);
        if !src.is_empty() {
            let mut device = buf.device.lock().unwrap();
            cuda_stream()
                .memcpy_htod(src, device.as_mut().expect("missing CUDA device buffer"))
                .expect("failed to copy host bytes to CUDA device buffer");
        }
    }

    fn len_bytes(buf: &Self::OwnedBuf) -> usize {
        buf.len
    }

    fn view(buf: &Self::OwnedBuf) -> Self::BufRef<'_> {
        CudaBufRef {
            ptr: NonNull::from(buf),
            offset: 0,
            len: buf.len,
            _marker: std::marker::PhantomData,
        }
    }

    fn view_ref<'a, 'b>(buf: &'a Self::BufRef<'b>) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        CudaBufRef {
            ptr: buf.ptr,
            offset: buf.offset,
            len: buf.len,
            _marker: std::marker::PhantomData,
        }
    }

    fn view_ref_mut<'a, 'b>(buf: &'a Self::BufMut<'b>) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        CudaBufRef {
            ptr: buf.ptr,
            offset: buf.offset,
            len: buf.len,
            _marker: std::marker::PhantomData,
        }
    }

    fn view_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>) -> Self::BufMut<'a>
    where
        Self: 'b,
    {
        CudaBufMut {
            ptr: buf.ptr,
            offset: buf.offset,
            len: buf.len,
            _marker: std::marker::PhantomData,
        }
    }

    fn view_mut(buf: &mut Self::OwnedBuf) -> Self::BufMut<'_> {
        CudaBufMut {
            ptr: NonNull::from(&mut *buf),
            offset: 0,
            len: buf.len,
            _marker: std::marker::PhantomData,
        }
    }

    fn region(buf: &Self::OwnedBuf, offset: usize, len: usize) -> Self::BufRef<'_> {
        CudaBufRef {
            ptr: NonNull::from(buf),
            offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    fn region_mut(buf: &mut Self::OwnedBuf, offset: usize, len: usize) -> Self::BufMut<'_> {
        CudaBufMut {
            ptr: NonNull::from(&mut *buf),
            offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    fn region_ref<'a, 'b>(buf: &'a Self::BufRef<'b>, offset: usize, len: usize) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        CudaBufRef {
            ptr: buf.ptr,
            offset: buf.offset + offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    fn region_ref_mut<'a, 'b>(buf: &'a Self::BufMut<'b>, offset: usize, len: usize) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        CudaBufRef {
            ptr: buf.ptr,
            offset: buf.offset + offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    fn region_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>, offset: usize, len: usize) -> Self::BufMut<'a>
    where
        Self: 'b,
    {
        CudaBufMut {
            ptr: buf.ptr,
            offset: buf.offset + offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe {
            drop(Box::from_raw(handle.as_ptr()));
        }
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use bytemuck::cast_slice;
    use poulpy_cpu_ref::{
        NTT120Ref,
        reference::ntt120::{PrimeSet, Primes30, Q120bScalar},
    };
    use poulpy_hal::{
        api::{
            SvpApplyDft, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPrepare, VecZnxDftAddAssign, VecZnxDftAlloc,
            VecZnxDftApply, VecZnxDftCopy, VecZnxDftZero,
        },
        layouts::{
            Backend, DataView, HostBytesBackend, MatZnx, Module, ScalarZnx, ScalarZnxToBackendRef, SvpPPolOwned,
            SvpPPolToBackendMut, SvpPPolToBackendRef, VecZnx, VecZnxDft, VecZnxDftToBackendMut, VecZnxDftToBackendRef,
            VecZnxToBackendRef,
        },
    };

    use super::CudaNtt120Backend;
    use crate::CudaBuf;

    fn is_cuda_unavailable(msg: &str) -> bool {
        msg.contains("failed to initialize CUDA context for device 0")
            || msg.contains("Unable to dynamically load the \"cuda\" shared library")
    }

    fn catch<F: FnOnce() + std::panic::UnwindSafe>(f: F) -> Result<(), String> {
        std::panic::catch_unwind(f).map_err(|e| {
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "unknown panic".to_string())
        })
    }

    fn alloc_host_vec_znx(n: usize, cols: usize, size: usize) -> VecZnx<Vec<u8>> {
        VecZnx::from_data(
            HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(n, cols, size)),
            n,
            cols,
            size,
        )
    }

    fn alloc_host_scalar_znx(n: usize, cols: usize) -> ScalarZnx<Vec<u8>> {
        ScalarZnx::from_data(
            HostBytesBackend::alloc_bytes(ScalarZnx::<Vec<u8>>::bytes_of(n, cols)),
            n,
            cols,
        )
    }

    fn alloc_host_mat_znx(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnx<Vec<u8>> {
        MatZnx::from_data(
            HostBytesBackend::alloc_bytes(MatZnx::<Vec<u8>>::bytes_of(n, rows, cols_in, cols_out, size)),
            n,
            rows,
            cols_in,
            cols_out,
            size,
        )
    }

    #[test]
    fn module_new_uploads_twiddles_without_panic() {
        if let Err(msg) = catch(|| {
            let _module = Module::<CudaNtt120Backend>::new(1 << 16);
        }) {
            if is_cuda_unavailable(&msg) {
                eprintln!("skipping NTT120 module construction test: {msg}");
            } else {
                panic!("{msg}");
            }
        }
    }

    /// Verify that GPU INTT(NTT(poly)) recovers the original residues.
    ///
    /// Tests the INTT kernels in isolation (before CRT): after a forward NTT followed
    /// by an in-place inverse NTT, each prime-major residue must equal the original
    /// polynomial coefficient reduced mod that prime, in the signed range (-Q/2, Q/2].
    #[test]
    fn ntt_inv_roundtrip() {
        const N: usize = 1 << 12;
        const COLS: usize = 2;
        const SIZE: usize = 3;

        let mut host: VecZnx<Vec<u8>> = alloc_host_vec_znx(N, COLS, SIZE);
        for (i, v) in bytemuck::cast_slice_mut::<u8, i64>(&mut host.data).iter_mut().enumerate() {
            *v = (i as i64 % 1013) - 506;
        }
        let host_i64: &[i64] = bytemuck::cast_slice(&host.data);

        let result = catch(std::panic::AssertUnwindSafe(|| {
            use cudarc::driver::DevicePtr;
            use poulpy_cpu_ref::reference::ntt120::{PrimeSet, Primes30};

            let module_gpu = Module::<CudaNtt120Backend>::new(N as u64);
            let handle = unsafe { &*module_gpu.as_mut_ptr() };
            let stream = crate::cuda_stream();
            let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

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
                fn ntt120_ntt_inv_apply(
                    stream: *mut std::ffi::c_void,
                    dst: *mut i32,
                    src: *const i32,
                    twiddle_inv: *const u32,
                    twiddle_inv_msb: *const u32,
                    primes: *const u32,
                    inv_primes: *const i32,
                    inv_n_mont: *const u32,
                    log_n: i32,
                    batch: i32,
                );
            }

            let get_u32 = |s: &cudarc::driver::CudaSlice<u32>| -> u64 {
                let (p, _g) = s.device_ptr(&stream);
                p
            };
            let get_i32 = |s: &cudarc::driver::CudaSlice<i32>| -> u64 {
                let (p, _g) = s.device_ptr(&stream);
                p
            };

            let gpu_src_buf = CudaNtt120Backend::from_host_bytes(&host.data);
            let dft_bytes = CudaNtt120Backend::bytes_of_vec_znx_dft(N, COLS, SIZE);
            let gpu_dft_buf = CudaNtt120Backend::alloc_bytes(dft_bytes);

            let src_ptr = gpu_src_buf.raw_device_ptr(0) as *const i64;
            let dft_ptr = gpu_dft_buf.raw_device_ptr(0) as *mut i32;
            let tw_fwd = get_u32(&handle.twiddle_fwd) as *const u32;
            let tw_fwd_msb = handle.twiddle_fwd_msb.as_ref().map(get_u32).unwrap_or(0) as *const u32;
            let tw_inv = get_u32(&handle.twiddle_inv) as *const u32;
            let tw_inv_msb = handle.twiddle_inv_msb.as_ref().map(get_u32).unwrap_or(0) as *const u32;
            let primes = get_u32(&handle.primes) as *const u32;
            let inv_primes = get_i32(&handle.inv_primes) as *const i32;
            let inv_n_mont = get_u32(&handle.inv_n_mont) as *const u32;
            let log_n = handle.log_n as i32;
            let batch = (COLS * SIZE) as i32;

            unsafe {
                ntt120_ntt_fwd_apply(
                    stream_raw, dft_ptr, src_ptr, tw_fwd, tw_fwd_msb, primes, inv_primes, log_n, batch,
                );
                ntt120_ntt_inv_apply(
                    stream_raw,
                    dft_ptr,
                    dft_ptr as *const i32,
                    tw_inv,
                    tw_inv_msb,
                    primes,
                    inv_primes,
                    inv_n_mont,
                    log_n,
                    batch,
                );
            }
            stream.synchronize().expect("CUDA sync failed");

            let out_bytes = CudaNtt120Backend::to_host_bytes(&gpu_dft_buf);
            let out_i32: &[i32] = bytemuck::cast_slice(&out_bytes);
            let primes_q = Primes30::Q;

            for col in 0..COLS {
                for limb in 0..SIZE {
                    let b = col * SIZE + limb;
                    for j in 0..N {
                        let coeff = host_i64[b * N + j];
                        for k in 0..4usize {
                            let q = primes_q[k] as i64;
                            let expected = ((coeff % q) + q) % q;
                            // MultConstNormalize maps to (-q/2, q/2]
                            let expected_signed = if expected > q / 2 { expected - q } else { expected };
                            let got = out_i32[(b * 4 + k) * N + j] as i64;
                            assert_eq!(
                                got, expected_signed,
                                "INTT roundtrip mismatch: col={col} limb={limb} j={j} k={k}: \
                                 got={got} expected={expected_signed} (coeff={coeff} q={q})"
                            );
                        }
                    }
                }
            }
        }));

        match result {
            Ok(()) => {}
            Err(msg) if is_cuda_unavailable(&msg) => {
                eprintln!("skipping ntt_inv_roundtrip: {msg}");
            }
            Err(msg) => panic!("{msg}"),
        }
    }

    /// Verify NTT → INTT+CRT round-trip: for a polynomial with small i64 coefficients,
    /// `vec_znx_idft_apply_tmpa(vec_znx_dft_apply(f))` must recover f as i128 Big32 values.
    #[test]
    fn idft_crt_roundtrip() {
        use poulpy_hal::{
            api::{VecZnxBigAlloc, VecZnxIdftApplyTmpA},
            layouts::{VecZnxBig, VecZnxBigToBackendMut},
        };

        const N: usize = 1 << 12;
        const COLS: usize = 2;
        const SIZE: usize = 3;

        let mut host: VecZnx<Vec<u8>> = alloc_host_vec_znx(N, COLS, SIZE);
        for (i, v) in bytemuck::cast_slice_mut::<u8, i64>(&mut host.data).iter_mut().enumerate() {
            *v = (i as i64 % 1013) - 506;
        }
        let host_i64: &[i64] = bytemuck::cast_slice(&host.data);

        let result = catch(std::panic::AssertUnwindSafe(|| {
            let module_gpu = Module::<CudaNtt120Backend>::new(N as u64);

            let gpu_src_buf = CudaNtt120Backend::from_host_bytes(&host.data);
            let gpu_src: VecZnx<CudaBuf> = VecZnx::from_data(gpu_src_buf, N, COLS, SIZE);

            let mut gpu_dft: VecZnxDft<CudaBuf, CudaNtt120Backend> = module_gpu.vec_znx_dft_alloc(COLS, SIZE);
            let mut gpu_big: VecZnxBig<CudaBuf, CudaNtt120Backend> = module_gpu.vec_znx_big_alloc(COLS, SIZE);

            for col in 0..COLS {
                let src_ref = <VecZnx<CudaBuf> as VecZnxToBackendRef<CudaNtt120Backend>>::to_backend_ref(&gpu_src);
                module_gpu.vec_znx_dft_apply(1, 0, &mut gpu_dft.to_backend_mut(), col, &src_ref, col);
            }

            for col in 0..COLS {
                module_gpu.vec_znx_idft_apply_tmpa(&mut gpu_big.to_backend_mut(), col, &mut gpu_dft.to_backend_mut(), col);
            }

            crate::cuda_stream().synchronize().expect("CUDA sync failed after IDFT+CRT");

            let big_bytes = CudaNtt120Backend::to_host_bytes(&gpu_big.data);
            let big_i128: &[i128] = bytemuck::cast_slice(&big_bytes);

            for col in 0..COLS {
                for limb in 0..SIZE {
                    let b = col * SIZE + limb;
                    for j in 0..N {
                        let got: i128 = big_i128[b * N + j];
                        let expected: i128 = host_i64[b * N + j] as i128;
                        assert_eq!(
                            got, expected,
                            "IDFT+CRT mismatch: col={col} limb={limb} j={j}: got={got} expected={expected}"
                        );
                    }
                }
            }
        }));

        match result {
            Ok(()) => {}
            Err(msg) if is_cuda_unavailable(&msg) => {
                eprintln!("skipping idft_crt_roundtrip: {msg}");
            }
            Err(msg) => panic!("{msg}"),
        }
    }

    /// Verify that the GPU forward NTT gives residues identical to the CPU reference.
    ///
    /// Layout comparison:
    ///   CPU (NTT120Ref): VecZnxDft → [cols × size × n] Q120bScalar ([u64; 4])
    ///     residue of NTT(poly_{col,limb})[j] mod Q[k] = cpu[(col*size+limb)*n + j].0[k]
    ///   GPU (CudaNtt120Backend): VecZnxDft → [cols × size × 4 × n] i32
    ///     residue of NTT(poly_{col,limb})[j] mod Q[k] = gpu[((col*size+limb)*4 + k)*n + j]
    #[test]
    fn ntt_fwd_parity_with_cpu_ref() {
        const N: usize = 1 << 12; // 4096 — smallest log_n supported by the kernel (case 12)
        const COLS: usize = 2;
        const SIZE: usize = 3;

        // Build a deterministic host polynomial with coefficients in [-500, 500].
        let mut host: VecZnx<Vec<u8>> = alloc_host_vec_znx(N, COLS, SIZE);
        for (i, v) in bytemuck::cast_slice_mut::<u8, i64>(&mut host.data).iter_mut().enumerate() {
            *v = (i as i64 % 1013) - 506;
        }

        // CPU: NTT120Ref forward NTT → Q120bScalar residues.
        let module_cpu = Module::<NTT120Ref>::new(N as u64);
        let mut cpu_dft: VecZnxDft<Vec<u8>, NTT120Ref> = module_cpu.vec_znx_dft_alloc(COLS, SIZE);
        for j in 0..COLS {
            let host_ref = <VecZnx<Vec<u8>> as VecZnxToBackendRef<NTT120Ref>>::to_backend_ref(&host);
            module_cpu.vec_znx_dft_apply(1, 0, &mut cpu_dft.to_backend_mut(), j, &host_ref, j);
        }

        // GPU: CudaNtt120Backend forward NTT → i32 residues (prime-major).
        let result = catch(std::panic::AssertUnwindSafe(|| {
            let module_gpu = Module::<CudaNtt120Backend>::new(N as u64);

            // Upload host polynomial to the GPU buffer.
            let gpu_src_buf: CudaBuf = CudaNtt120Backend::from_host_bytes(&host.data);
            let gpu_src: VecZnx<CudaBuf> = VecZnx::from_data(gpu_src_buf, N, COLS, SIZE);

            let mut gpu_dft: VecZnxDft<CudaBuf, CudaNtt120Backend> = module_gpu.vec_znx_dft_alloc(COLS, SIZE);

            for col in 0..COLS {
                let src_ref = <VecZnx<CudaBuf> as VecZnxToBackendRef<CudaNtt120Backend>>::to_backend_ref(&gpu_src);
                module_gpu.vec_znx_dft_apply(1, 0, &mut gpu_dft.to_backend_mut(), col, &src_ref, col);
            }
            crate::cuda_stream()
                .synchronize()
                .expect("CUDA stream sync failed after NTT kernel");

            // Download GPU output and compare residue by residue.
            let gpu_bytes: Vec<u8> = CudaNtt120Backend::to_host_bytes(&gpu_dft.data);
            let gpu_i32: &[i32] = cast_slice(&gpu_bytes);

            let cpu_bytes: &[u8] = cpu_dft.data.as_slice();
            let cpu_scalars: &[Q120bScalar] = cast_slice(cpu_bytes);

            let primes = Primes30::Q;
            for col in 0..COLS {
                for limb in 0..SIZE {
                    let batch = col * SIZE + limb;
                    for j in 0..N {
                        let cpu_s = &cpu_scalars[batch * N + j];
                        for k in 0..4usize {
                            let cpu_val = (cpu_s.0[k] % primes[k] as u64) as u32;
                            let gpu_raw = gpu_i32[(batch * 4 + k) * N + j];
                            assert!(
                                gpu_raw >= 0,
                                "GPU residue negative: col={col} limb={limb} j={j} k={k} val={gpu_raw}"
                            );
                            let gpu_val = gpu_raw as u32;
                            assert_eq!(
                                gpu_val, cpu_val,
                                "NTT mismatch: col={col} limb={limb} coeff={j} prime_k={k}: gpu={gpu_val} cpu={cpu_val}"
                            );
                        }
                    }
                }
            }
        }));

        match result {
            Ok(()) => {}
            Err(msg) if is_cuda_unavailable(&msg) => {
                eprintln!("skipping ntt_fwd_parity_with_cpu_ref: {msg}");
            }
            Err(msg) => panic!("{msg}"),
        }
    }

    /// Verify that `ntt120_big_normalize` matches the CPU reference (inter case, COLS=1).
    ///
    /// For each (a_size, res_size) pair and each res_offset in [-base2k, base2k],
    /// normalizes a random i128 input on GPU and checks against an inlined CPU reference.
    #[test]
    fn big_normalize_parity_with_cpu_ref() {
        const N: usize = 128;
        const BASE2K: usize = 50;

        fn get_digit(x: i128, base2k: usize) -> i128 {
            let shift = 128 - base2k;
            (x << shift) >> shift
        }
        fn get_carry(x: i128, digit: i128, base2k: usize) -> i128 {
            (x.wrapping_sub(digit)) >> base2k
        }

        fn cpu_normalize(a: &[i128], a_size: usize, res_size: usize, base2k: usize, res_offset: i64, n: usize) -> Vec<i64> {
            let mut lsh = res_offset % base2k as i64;
            let mut limbs_offset = res_offset / base2k as i64;
            if res_offset < 0 && lsh != 0 {
                lsh = (lsh + base2k as i64) % base2k as i64;
                limbs_offset -= 1;
            }
            let lsh = lsh as usize;
            let base2k_lsh = base2k - lsh;

            let res_end = (-limbs_offset).clamp(0, res_size as i64) as usize;
            let res_start = (a_size as i64 - limbs_offset).clamp(0, res_size as i64) as usize;
            let a_end = limbs_offset.clamp(0, a_size as i64) as usize;
            let a_start = (res_size as i64 + limbs_offset).clamp(0, a_size as i64) as usize;
            let a_out_range = a_size - a_start;
            let mid_range = a_start - a_end;

            let mut result = vec![0i64; res_size * n];

            for j in 0..n {
                let mut carry = 0i128;

                for k in 0..a_out_range {
                    let limb = a_size - k - 1;
                    let ai = a[limb * n + j];
                    if k == 0 {
                        if lsh == 0 {
                            let d = get_digit(ai, base2k);
                            carry = get_carry(ai, d, base2k);
                        } else {
                            let d = get_digit(ai, base2k_lsh);
                            carry = get_carry(ai, d, base2k_lsh);
                        }
                    } else if lsh == 0 {
                        let d = get_digit(ai, base2k);
                        let co = get_carry(ai, d, base2k);
                        let dc = d + carry;
                        carry = co + get_carry(dc, get_digit(dc, base2k), base2k);
                    } else {
                        let d = get_digit(ai, base2k_lsh);
                        let co = get_carry(ai, d, base2k_lsh);
                        let dc = (d << lsh) + carry;
                        carry = co + get_carry(dc, get_digit(dc, base2k), base2k);
                    }
                }
                if a_out_range == 0 {
                    carry = 0;
                }

                for k in res_start..res_size {
                    result[k * n + j] = 0;
                }

                for k in 0..mid_range {
                    let a_limb = a_start - k - 1;
                    let r_limb = res_start - k - 1;
                    let ai = a[a_limb * n + j];
                    let out128 = if lsh == 0 {
                        let d = get_digit(ai, base2k);
                        let co = get_carry(ai, d, base2k);
                        let dc = d + carry;
                        let out = get_digit(dc, base2k);
                        carry = co + get_carry(dc, out, base2k);
                        out
                    } else {
                        let d = get_digit(ai, base2k_lsh);
                        let co = get_carry(ai, d, base2k_lsh);
                        let dc = (d << lsh) + carry;
                        let out = get_digit(dc, base2k);
                        carry = co + get_carry(dc, out, base2k);
                        out
                    };
                    result[r_limb * n + j] = out128 as i64;
                }

                for k in 0..res_end {
                    let r_limb = res_end - k - 1;
                    if k == res_end - 1 {
                        result[r_limb * n + j] = get_digit(carry, base2k) as i64;
                    } else {
                        let out128 = get_digit(carry, base2k);
                        result[r_limb * n + j] = out128 as i64;
                        carry = get_carry(carry, out128, base2k);
                    }
                }
            }
            result
        }

        let result = catch(std::panic::AssertUnwindSafe(|| {
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
            }

            let stream = crate::cuda_stream();
            let stream_raw = stream.cu_stream() as *mut std::ffi::c_void;

            // Deterministic i128 inputs with values in (-2^BASE2K*4, 2^BASE2K*4).
            let max_a_size: usize = 4;
            let mut a_vals: Vec<i128> = (0..max_a_size * N)
                .map(|i| {
                    let v = (i as i128 * 123456789 + 987654321) % (1i128 << (BASE2K + 2));
                    if i % 3 == 0 { -v } else { v }
                })
                .collect();
            // Ensure some zero entries.
            a_vals[0] = 0;
            a_vals[N / 2] = 1i128 << (BASE2K - 1);

            let src_bytes: Vec<u8> = bytemuck::cast_slice::<i128, u8>(&a_vals).to_vec();
            let gpu_src_buf = CudaNtt120Backend::from_host_bytes(&src_bytes);
            let src_ptr = gpu_src_buf.raw_device_ptr(0);

            let max_res_size: usize = 4;
            let res_buf_bytes = max_res_size * N * std::mem::size_of::<i64>();
            let gpu_res_buf = CudaNtt120Backend::alloc_bytes(res_buf_bytes);
            let res_ptr = gpu_res_buf.raw_device_ptr(0);

            for a_size in 1..=max_a_size {
                for res_size in 1..=max_res_size {
                    for res_offset in -(BASE2K as i64)..=(BASE2K as i64) {
                        // GPU normalize.
                        unsafe {
                            ntt120_big_normalize(
                                stream_raw,
                                res_ptr as *mut i64,
                                src_ptr as *const u32,
                                N as i32,
                                a_size as i32,
                                res_size as i32,
                                BASE2K as i32,
                                res_offset,
                            );
                        }
                        stream.synchronize().expect("CUDA sync failed after big_normalize");

                        let gpu_bytes = CudaNtt120Backend::to_host_bytes(&gpu_res_buf);
                        let gpu_res: &[i64] = bytemuck::cast_slice(&gpu_bytes[..res_size * N * 8]);

                        // CPU reference.
                        let cpu_res = cpu_normalize(&a_vals[..a_size * N], a_size, res_size, BASE2K, res_offset, N);

                        assert_eq!(
                            gpu_res,
                            cpu_res.as_slice(),
                            "mismatch: a_size={a_size} res_size={res_size} res_offset={res_offset}"
                        );
                    }
                }
            }
        }));

        match result {
            Ok(()) => {}
            Err(msg) if is_cuda_unavailable(&msg) => {
                eprintln!("skipping big_normalize_parity_with_cpu_ref: {msg}");
            }
            Err(msg) => panic!("{msg}"),
        }
    }

    /// Test `vec_znx_dft_zero`, `vec_znx_dft_copy`, and `vec_znx_dft_add_assign`.
    ///
    /// - Zero: a forward-NTT'd buffer zeroed on GPU must be all-zero bytes.
    /// - Copy: a buffer copied from a DFT column must match the source byte-for-byte.
    /// - Add:  GPU `b_dft += a_dft` must match `NTT(a)[k] + NTT(b)[k] mod Q[k]`
    ///   computed from the CPU reference DFT outputs.
    #[test]
    fn dft_zero_copy_add_assign_parity() {
        const N: usize = 1 << 12;
        const COLS: usize = 1;
        const SIZE: usize = 2;

        // Deterministic host polynomials with small signed coefficients.
        let mut a_host: VecZnx<Vec<u8>> = alloc_host_vec_znx(N, COLS, SIZE);
        let mut b_host: VecZnx<Vec<u8>> = alloc_host_vec_znx(N, COLS, SIZE);
        for (i, v) in bytemuck::cast_slice_mut::<u8, i64>(&mut a_host.data).iter_mut().enumerate() {
            *v = (i as i64 % 997) - 498;
        }
        for (i, v) in bytemuck::cast_slice_mut::<u8, i64>(&mut b_host.data).iter_mut().enumerate() {
            *v = (i as i64 % 503) - 251;
        }

        // CPU reference forward NTTs.
        let module_cpu = Module::<NTT120Ref>::new(N as u64);
        let mut a_dft_cpu: VecZnxDft<Vec<u8>, NTT120Ref> = module_cpu.vec_znx_dft_alloc(COLS, SIZE);
        let mut b_dft_cpu: VecZnxDft<Vec<u8>, NTT120Ref> = module_cpu.vec_znx_dft_alloc(COLS, SIZE);
        module_cpu.vec_znx_dft_apply(
            1,
            0,
            &mut a_dft_cpu.to_backend_mut(),
            0,
            &<VecZnx<Vec<u8>> as VecZnxToBackendRef<NTT120Ref>>::to_backend_ref(&a_host),
            0,
        );
        module_cpu.vec_znx_dft_apply(
            1,
            0,
            &mut b_dft_cpu.to_backend_mut(),
            0,
            &<VecZnx<Vec<u8>> as VecZnxToBackendRef<NTT120Ref>>::to_backend_ref(&b_host),
            0,
        );

        let result = catch(std::panic::AssertUnwindSafe(|| {
            let module_gpu = Module::<CudaNtt120Backend>::new(N as u64);

            // Upload host polynomials.
            let a_gpu_src = CudaNtt120Backend::from_host_bytes(&a_host.data);
            let b_gpu_src = CudaNtt120Backend::from_host_bytes(&b_host.data);
            let a_vec: VecZnx<CudaBuf> = VecZnx::from_data(a_gpu_src, N, COLS, SIZE);
            let b_vec: VecZnx<CudaBuf> = VecZnx::from_data(b_gpu_src, N, COLS, SIZE);

            // GPU forward NTT.
            let mut a_dft: VecZnxDft<CudaBuf, CudaNtt120Backend> = module_gpu.vec_znx_dft_alloc(COLS, SIZE);
            let mut b_dft: VecZnxDft<CudaBuf, CudaNtt120Backend> = module_gpu.vec_znx_dft_alloc(COLS, SIZE);
            module_gpu.vec_znx_dft_apply(
                1,
                0,
                &mut a_dft.to_backend_mut(),
                0,
                &<VecZnx<CudaBuf> as VecZnxToBackendRef<CudaNtt120Backend>>::to_backend_ref(&a_vec),
                0,
            );
            module_gpu.vec_znx_dft_apply(
                1,
                0,
                &mut b_dft.to_backend_mut(),
                0,
                &<VecZnx<CudaBuf> as VecZnxToBackendRef<CudaNtt120Backend>>::to_backend_ref(&b_vec),
                0,
            );

            // ── vec_znx_dft_zero ──────────────────────────────────────────────
            let mut tmp: VecZnxDft<CudaBuf, CudaNtt120Backend> = module_gpu.vec_znx_dft_alloc(COLS, SIZE);
            // Populate tmp by copying a_dft (so we know it has non-zero data).
            module_gpu.vec_znx_dft_copy(
                1,
                0,
                &mut tmp.to_backend_mut(),
                0,
                &<VecZnxDft<CudaBuf, CudaNtt120Backend> as VecZnxDftToBackendRef<CudaNtt120Backend>>::to_backend_ref(&a_dft),
                0,
            );
            module_gpu.vec_znx_dft_zero(&mut tmp.to_backend_mut(), 0);
            crate::cuda_stream().synchronize().expect("sync after zero");
            let zero_bytes = CudaNtt120Backend::to_host_bytes(&tmp.data);
            assert!(zero_bytes.iter().all(|&b| b == 0), "vec_znx_dft_zero: buffer not zeroed");

            // ── vec_znx_dft_copy ──────────────────────────────────────────────
            let mut copy_dft: VecZnxDft<CudaBuf, CudaNtt120Backend> = module_gpu.vec_znx_dft_alloc(COLS, SIZE);
            module_gpu.vec_znx_dft_copy(
                1,
                0,
                &mut copy_dft.to_backend_mut(),
                0,
                &<VecZnxDft<CudaBuf, CudaNtt120Backend> as VecZnxDftToBackendRef<CudaNtt120Backend>>::to_backend_ref(&a_dft),
                0,
            );
            crate::cuda_stream().synchronize().expect("sync after copy");
            let a_bytes = CudaNtt120Backend::to_host_bytes(&a_dft.data);
            let copy_bytes = CudaNtt120Backend::to_host_bytes(&copy_dft.data);
            assert_eq!(a_bytes, copy_bytes, "vec_znx_dft_copy: data mismatch");

            // ── vec_znx_dft_add_assign ────────────────────────────────────────
            // b_dft += a_dft
            module_gpu.vec_znx_dft_add_assign(
                &mut b_dft.to_backend_mut(),
                0,
                &<VecZnxDft<CudaBuf, CudaNtt120Backend> as VecZnxDftToBackendRef<CudaNtt120Backend>>::to_backend_ref(&a_dft),
                0,
            );
            crate::cuda_stream().synchronize().expect("sync after add_assign");

            let gpu_sum_bytes = CudaNtt120Backend::to_host_bytes(&b_dft.data);
            let gpu_sum: &[i32] = cast_slice(&gpu_sum_bytes);
            let a_cpu: &[Q120bScalar] = cast_slice(a_dft_cpu.data.as_slice());
            let b_cpu: &[Q120bScalar] = cast_slice(b_dft_cpu.data.as_slice());
            let primes = Primes30::Q;

            for limb in 0..SIZE {
                for j in 0..N {
                    for k in 0..4usize {
                        let pa = (a_cpu[limb * N + j].0[k] % primes[k] as u64) as u32;
                        let pb = (b_cpu[limb * N + j].0[k] % primes[k] as u64) as u32;
                        let expected = if pa + pb >= primes[k] { pa + pb - primes[k] } else { pa + pb };
                        let got = gpu_sum[(limb * 4 + k) * N + j] as u32;
                        assert_eq!(
                            got, expected,
                            "add_assign mismatch: limb={limb} j={j} k={k}: got={got} expected={expected}"
                        );
                    }
                }
            }
        }));

        match result {
            Ok(()) => {}
            Err(msg) if is_cuda_unavailable(&msg) => {
                eprintln!("skipping dft_zero_copy_add_assign_parity: {msg}");
            }
            Err(msg) => panic!("{msg}"),
        }
    }

    /// Verify that GPU `svp_prepare` + `svp_apply_dft` / `svp_apply_dft_to_dft` /
    /// `svp_apply_dft_to_dft_inplace` all match the CPU reference (NTT120Ref).
    ///
    /// Comparison is residue-by-residue: for each limb, coefficient j, and prime k,
    /// `gpu_i32[(limb * 4 + k) * N + j]` must equal
    /// `cpu_scalar[limb * N + j].0[k] % Q[k]`.
    #[test]
    fn svp_parity_with_cpu_ref() {
        const N: usize = 1 << 12;
        const COLS: usize = 2;
        const SIZE: usize = 3;

        // Deterministic scalar polynomial (1 col, 1 limb = ScalarZnx) and vector.
        let mut scalar_host: ScalarZnx<Vec<u8>> = alloc_host_scalar_znx(N, 1);
        let mut b_host: VecZnx<Vec<u8>> = alloc_host_vec_znx(N, COLS, SIZE);
        for (i, v) in bytemuck::cast_slice_mut::<u8, i64>(&mut scalar_host.data)
            .iter_mut()
            .enumerate()
        {
            *v = (i as i64 % 997) - 498;
        }
        for (i, v) in bytemuck::cast_slice_mut::<u8, i64>(&mut b_host.data).iter_mut().enumerate() {
            *v = (i as i64 % 503) - 251;
        }

        // ── CPU reference ─────────────────────────────────────────────────────
        let module_cpu = Module::<NTT120Ref>::new(N as u64);

        // Prepare scalar for NTT120Ref.
        let scalar_cpu = ScalarZnx::from_data(NTT120Ref::from_host_bytes(scalar_host.data.as_slice()), N, 1);
        let mut svp_cpu: SvpPPolOwned<NTT120Ref> = module_cpu.svp_ppol_alloc(1);
        module_cpu.svp_prepare(
            &mut svp_cpu.to_backend_mut(),
            0,
            &<ScalarZnx<Vec<u8>> as ScalarZnxToBackendRef<NTT120Ref>>::to_backend_ref(&scalar_cpu),
            0,
        );

        // svp_apply_dft: NTT(b) * scalar → cpu_dft.
        let mut cpu_dft: VecZnxDft<Vec<u8>, NTT120Ref> = module_cpu.vec_znx_dft_alloc(COLS, SIZE);
        for col in 0..COLS {
            module_cpu.svp_apply_dft(
                &mut cpu_dft.to_backend_mut(),
                col,
                &svp_cpu.to_backend_ref(),
                0,
                &<VecZnx<Vec<u8>> as VecZnxToBackendRef<NTT120Ref>>::to_backend_ref(&b_host),
                col,
            );
        }

        // svp_apply_dft_to_dft: cpu_dft * scalar → cpu_dft2.
        let mut cpu_dft2: VecZnxDft<Vec<u8>, NTT120Ref> = module_cpu.vec_znx_dft_alloc(COLS, SIZE);
        for col in 0..COLS {
            module_cpu.svp_apply_dft_to_dft(
                &mut cpu_dft2.to_backend_mut(),
                col,
                &svp_cpu.to_backend_ref(),
                0,
                &<VecZnxDft<Vec<u8>, NTT120Ref> as VecZnxDftToBackendRef<NTT120Ref>>::to_backend_ref(&cpu_dft),
                col,
            );
        }

        // svp_apply_dft_to_dft_inplace: cpu_dft2 *= scalar.
        let mut cpu_dft3: VecZnxDft<Vec<u8>, NTT120Ref> = module_cpu.vec_znx_dft_alloc(COLS, SIZE);
        for col in 0..COLS {
            module_cpu.vec_znx_dft_copy(
                1,
                0,
                &mut cpu_dft3.to_backend_mut(),
                col,
                &<VecZnxDft<Vec<u8>, NTT120Ref> as VecZnxDftToBackendRef<NTT120Ref>>::to_backend_ref(&cpu_dft2),
                col,
            );
            module_cpu.svp_apply_dft_to_dft_inplace(&mut cpu_dft3.to_backend_mut(), col, &svp_cpu.to_backend_ref(), 0);
        }

        // ── GPU ───────────────────────────────────────────────────────────────
        let result = catch(std::panic::AssertUnwindSafe(|| {
            let module_gpu = Module::<CudaNtt120Backend>::new(N as u64);

            // Prepare scalar on GPU.
            let scalar_gpu = ScalarZnx::from_data(CudaNtt120Backend::from_host_bytes(scalar_host.data.as_slice()), N, 1);
            let mut svp_gpu: SvpPPolOwned<CudaNtt120Backend> = module_gpu.svp_ppol_alloc(1);
            module_gpu.svp_prepare(
                &mut svp_gpu.to_backend_mut(),
                0,
                &<ScalarZnx<CudaBuf> as ScalarZnxToBackendRef<CudaNtt120Backend>>::to_backend_ref(&scalar_gpu),
                0,
            );

            // Upload b to GPU.
            let b_gpu = CudaNtt120Backend::from_host_bytes(&b_host.data);
            let b_vec: VecZnx<CudaBuf> = VecZnx::from_data(b_gpu, N, COLS, SIZE);

            let compare_dft =
                |gpu_dft: &VecZnxDft<CudaBuf, CudaNtt120Backend>, cpu_dft: &VecZnxDft<Vec<u8>, NTT120Ref>, label: &str| {
                    crate::cuda_stream().synchronize().expect("CUDA sync failed");
                    let gpu_bytes = CudaNtt120Backend::to_host_bytes(&gpu_dft.data);
                    let gpu_i32: &[i32] = cast_slice(&gpu_bytes);
                    let cpu_scalars: &[Q120bScalar] = cast_slice(cpu_dft.data.as_slice());
                    let primes = Primes30::Q;
                    for col in 0..COLS {
                        for limb in 0..SIZE {
                            let b = col * SIZE + limb;
                            for j in 0..N {
                                let cpu_s = &cpu_scalars[b * N + j];
                                for k in 0..4usize {
                                    let cpu_val = (cpu_s.0[k] % primes[k] as u64) as u32;
                                    let gpu_raw = gpu_i32[(b * 4 + k) * N + j];
                                    assert!(
                                        gpu_raw >= 0,
                                        "{label}: GPU residue negative: col={col} limb={limb} j={j} k={k} val={gpu_raw}"
                                    );
                                    let gpu_val = gpu_raw as u32;
                                    assert_eq!(
                                        gpu_val, cpu_val,
                                        "{label}: mismatch: col={col} limb={limb} j={j} k={k}: gpu={gpu_val} cpu={cpu_val}"
                                    );
                                }
                            }
                        }
                    }
                };

            // svp_apply_dft parity.
            let mut gpu_dft: VecZnxDft<CudaBuf, CudaNtt120Backend> = module_gpu.vec_znx_dft_alloc(COLS, SIZE);
            for col in 0..COLS {
                module_gpu.svp_apply_dft(
                    &mut gpu_dft.to_backend_mut(),
                    col,
                    &svp_gpu.to_backend_ref(),
                    0,
                    &<VecZnx<CudaBuf> as VecZnxToBackendRef<CudaNtt120Backend>>::to_backend_ref(&b_vec),
                    col,
                );
            }
            compare_dft(&gpu_dft, &cpu_dft, "svp_apply_dft");

            // svp_apply_dft_to_dft parity.
            let mut gpu_dft2: VecZnxDft<CudaBuf, CudaNtt120Backend> = module_gpu.vec_znx_dft_alloc(COLS, SIZE);
            for col in 0..COLS {
                module_gpu.svp_apply_dft_to_dft(
                    &mut gpu_dft2.to_backend_mut(),
                    col,
                    &svp_gpu.to_backend_ref(),
                    0,
                    &<VecZnxDft<CudaBuf, CudaNtt120Backend> as VecZnxDftToBackendRef<CudaNtt120Backend>>::to_backend_ref(
                        &gpu_dft,
                    ),
                    col,
                );
            }
            compare_dft(&gpu_dft2, &cpu_dft2, "svp_apply_dft_to_dft");

            // svp_apply_dft_to_dft_inplace parity.
            let mut gpu_dft3: VecZnxDft<CudaBuf, CudaNtt120Backend> = module_gpu.vec_znx_dft_alloc(COLS, SIZE);
            for col in 0..COLS {
                module_gpu.vec_znx_dft_copy(
                    1,
                    0,
                    &mut gpu_dft3.to_backend_mut(),
                    col,
                    &<VecZnxDft<CudaBuf, CudaNtt120Backend> as VecZnxDftToBackendRef<CudaNtt120Backend>>::to_backend_ref(
                        &gpu_dft2,
                    ),
                    col,
                );
                module_gpu.svp_apply_dft_to_dft_inplace(&mut gpu_dft3.to_backend_mut(), col, &svp_gpu.to_backend_ref(), 0);
            }
            compare_dft(&gpu_dft3, &cpu_dft3, "svp_apply_dft_to_dft_inplace");
        }));

        match result {
            Ok(()) => {}
            Err(msg) if is_cuda_unavailable(&msg) => {
                eprintln!("skipping svp_parity_with_cpu_ref: {msg}");
            }
            Err(msg) => panic!("{msg}"),
        }
    }

    /// Verify `vmp_prepare` + `vmp_apply_dft` parity between GPU and CPU NTT120Ref.
    ///
    /// For each (rows, cols_in, cols_out, size_in, size_out) combination:
    ///   1. Random `MatZnx` → `vmp_prepare` on both backends.
    ///   2. Random `VecZnx` → `vmp_apply_dft` on both backends.
    ///   3. Compare the DFT output residues (mod each of the 4 primes) exactly.
    #[test]
    fn vmp_parity_with_cpu_ref() {
        use poulpy_hal::{
            api::{
                ScratchOwnedAlloc, VecZnxDftAlloc, VmpApplyDft, VmpApplyDftTmpBytes, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
            },
            layouts::{FillUniform, MatZnx, MatZnxToBackendRef, ScratchOwned, VmpPMatToBackendMut, VmpPMatToBackendRef},
            source::Source,
        };

        const N: usize = 1 << 12;
        const MAX_ROWS: usize = 2;
        const MAX_COLS_IN: usize = 2;
        const MAX_COLS_OUT: usize = 2;
        const MAX_SIZE: usize = 3;
        const BASE2K: usize = 30;

        let mut source = Source::new([42u8; 32]);

        let module_cpu = Module::<NTT120Ref>::new(N as u64);
        let primes = Primes30::Q;

        let result = catch(std::panic::AssertUnwindSafe(|| {
            let module_gpu = Module::<CudaNtt120Backend>::new(N as u64);

            for rows in 1..=MAX_ROWS {
                for cols_in in 1..=MAX_COLS_IN {
                    for cols_out in 1..=MAX_COLS_OUT {
                        for size_in in 1..=MAX_SIZE {
                            for size_out in 1..=MAX_SIZE {
                                // ── allocate scratch ─────────────────────────────────
                                let prepare_bytes_cpu = module_cpu.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size_out);
                                let prepare_bytes_gpu = module_gpu.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size_out);
                                let apply_bytes_cpu =
                                    module_cpu.vmp_apply_dft_tmp_bytes(size_out, size_in, rows, cols_in, cols_out, size_out);
                                let apply_bytes_gpu =
                                    module_gpu.vmp_apply_dft_tmp_bytes(size_out, size_in, rows, cols_in, cols_out, size_out);
                                let mut scratch_cpu: ScratchOwned<NTT120Ref> =
                                    ScratchOwned::alloc(prepare_bytes_cpu.max(apply_bytes_cpu));
                                let mut scratch_gpu: ScratchOwned<CudaNtt120Backend> =
                                    ScratchOwned::alloc(prepare_bytes_gpu.max(apply_bytes_gpu));

                                // ── random MatZnx ─────────────────────────────────────
                                let mut mat = alloc_host_mat_znx(N, rows, cols_in, cols_out, size_out);
                                mat.fill_uniform(BASE2K, &mut source);

                                // ── random VecZnx input ───────────────────────────────
                                let mut a_host: VecZnx<Vec<u8>> = alloc_host_vec_znx(N, cols_in, size_in);
                                a_host.fill_uniform(BASE2K, &mut source);

                                // ── CPU prepare + apply ───────────────────────────────
                                let mut pmat_cpu = module_cpu.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);
                                module_cpu.vmp_prepare(
                                    &mut pmat_cpu.to_backend_mut(),
                                    &<MatZnx<Vec<u8>> as MatZnxToBackendRef<NTT120Ref>>::to_backend_ref(&mat),
                                    &mut scratch_cpu.arena(),
                                );
                                let mut res_cpu: VecZnxDft<Vec<u8>, NTT120Ref> = module_cpu.vec_znx_dft_alloc(cols_out, size_out);
                                module_cpu.vmp_apply_dft(
                                    &mut res_cpu,
                                    &<VecZnx<Vec<u8>> as VecZnxToBackendRef<NTT120Ref>>::to_backend_ref(&a_host),
                                    &pmat_cpu.to_backend_ref(),
                                    &mut scratch_cpu.arena(),
                                );

                                // ── GPU prepare + apply ───────────────────────────────
                                let mut pmat_gpu = module_gpu.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);
                                let mat_gpu: MatZnx<CudaBuf> = MatZnx::from_data(
                                    CudaNtt120Backend::from_host_bytes(mat.data()),
                                    N,
                                    rows,
                                    cols_in,
                                    cols_out,
                                    size_out,
                                );
                                module_gpu.vmp_prepare(
                                    &mut pmat_gpu.to_backend_mut(),
                                    &<MatZnx<CudaBuf> as MatZnxToBackendRef<CudaNtt120Backend>>::to_backend_ref(&mat_gpu),
                                    &mut scratch_gpu.arena(),
                                );
                                let a_gpu = CudaNtt120Backend::from_host_bytes(&a_host.data);
                                let a_vec: VecZnx<CudaBuf> = VecZnx::from_data(a_gpu, N, cols_in, size_in);
                                let mut res_gpu: VecZnxDft<CudaBuf, CudaNtt120Backend> =
                                    module_gpu.vec_znx_dft_alloc(cols_out, size_out);
                                module_gpu.vmp_apply_dft(
                                    &mut res_gpu,
                                    &<VecZnx<CudaBuf> as VecZnxToBackendRef<CudaNtt120Backend>>::to_backend_ref(&a_vec),
                                    &pmat_gpu.to_backend_ref(),
                                    &mut scratch_gpu.arena(),
                                );
                                crate::cuda_stream().synchronize().expect("CUDA sync failed in vmp test");

                                // ── compare residues ──────────────────────────────────
                                let gpu_bytes = CudaNtt120Backend::to_host_bytes(&res_gpu.data);
                                let gpu_i32: &[i32] = cast_slice(&gpu_bytes);
                                let cpu_scalars: &[Q120bScalar] = cast_slice(res_cpu.data.as_slice());
                                let total_limbs = cols_out * size_out;
                                for b in 0..total_limbs {
                                    for j in 0..N {
                                        for k in 0..4usize {
                                            let cpu_val = (cpu_scalars[b * N + j].0[k] % primes[k] as u64) as u32;
                                            let gpu_raw = gpu_i32[(b * 4 + k) * N + j];
                                            assert!(
                                                gpu_raw >= 0,
                                                "GPU residue negative: rows={rows} ci={cols_in} co={cols_out} si={size_in} so={size_out} limb={b} j={j} k={k} val={gpu_raw}"
                                            );
                                            let gpu_val = gpu_raw as u32;
                                            assert_eq!(
                                                gpu_val, cpu_val,
                                                "VMP mismatch: rows={rows} ci={cols_in} co={cols_out} si={size_in} so={size_out} limb={b} j={j} k={k}: gpu={gpu_val} cpu={cpu_val}"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }));

        match result {
            Ok(()) => {}
            Err(msg) if is_cuda_unavailable(&msg) => {
                eprintln!("skipping vmp_parity_with_cpu_ref: {msg}");
            }
            Err(msg) => panic!("{msg}"),
        }
    }
}

unsafe impl HalModuleImpl<CudaNtt120Backend> for CudaNtt120Backend {
    fn new(n: u64) -> Module<CudaNtt120Backend> {
        let _ = crate::cuda_context();
        let log_n = n.trailing_zeros() as usize;
        let tables = twiddle::generate(n as usize);
        let stream = cuda_stream();

        let upload = |v: &[u32]| -> CudaSlice<u32> { stream.clone_htod(v).expect("failed to upload twiddle table") };
        let upload_opt = |v: &[u32]| -> Option<CudaSlice<u32>> { if v.is_empty() { None } else { Some(upload(v)) } };

        let handle = CudaNtt120Handle {
            twiddle_fwd: upload(&tables.twiddle_fwd),
            twiddle_fwd_msb: upload_opt(&tables.twiddle_fwd_msb),
            twiddle_inv: upload(&tables.twiddle_inv),
            twiddle_inv_msb: upload_opt(&tables.twiddle_inv_msb),
            inv_n_mont: upload(&tables.inv_n_mont),
            primes: upload(&tables.primes),
            inv_primes: stream.clone_htod(&tables.inv_primes).expect("failed to upload inv_primes"),
            log_n,
        };
        let ptr = NonNull::from(Box::leak(Box::new(handle)));
        unsafe { Module::from_nonnull(ptr, n) }
    }
}
