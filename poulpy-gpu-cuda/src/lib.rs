//! Minimal CUDA-backed backend prototype for Poulpy.
//!
//! This crate is intentionally narrow and is meant to document the smallest
//! useful pattern for an external device backend:
//! - a distinct `Device` backend type
//! - an owned buffer type that keeps device memory plus a host mirror
//! - explicit CPU/CUDA transfers
//! - backend-native scratch allocation and typed arena carving
//!
//! The host mirror is still present because large parts of the current HAL
//! surface remain host-readable. The current CUDA slice focuses on owned
//! buffers plus `ScratchArena` carving over CUDA-backed storage.

use std::{
    fmt,
    ptr::NonNull,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use poulpy_cpu_ref::reference::fft64::{
    module::{FFT64HandleFactory, FFTHandleProvider},
    reim::{ReimFFTTable, ReimIFFTTable},
};
use poulpy_hal::{
    alloc_aligned, assert_alignment,
    layouts::{Backend, Device, TransferFrom},
};

mod hal_impl;

#[cfg(test)]
mod tests;

/// Minimal CUDA-backed `Device` backend used to document external-device integration.
#[derive(Debug, Clone, Copy)]
pub struct CudaGpuBackend;

/// FFT tables owned by the CUDA backend's module handle.
///
/// The backend still reuses the existing FFT64 host-side tables for module
/// setup because the current CUDA slice is focused on backend-owned `VecZnx`
/// operations rather than DFT-domain execution.
#[repr(C)]
pub struct CudaFft64Handle {
    table_fft: ReimFFTTable<f64>,
    table_ifft: ReimIFFTTable<f64>,
}

/// Backend-owned byte buffer for CUDA layouts.
///
/// `host` remains the source of truth for host-readable HAL paths. `device`
/// holds the CUDA allocation mirrored by backend-native scratch/storage.
pub struct CudaBuf {
    host: Vec<u8>,
    device: Mutex<Option<CudaSlice<u8>>>,
    device_stale: AtomicBool,
}

impl Default for CudaBuf {
    fn default() -> Self {
        Self {
            host: Vec::new(),
            device: Mutex::new(None),
            device_stale: AtomicBool::new(false),
        }
    }
}

/// Shared backend-native view into a CUDA buffer.
#[derive(Clone, Copy)]
pub struct CudaBufRef<'a> {
    pub(crate) ptr: NonNull<CudaBuf>,
    pub(crate) offset: usize,
    pub(crate) len: usize,
    pub(crate) _marker: std::marker::PhantomData<&'a CudaBuf>,
}

// SAFETY: `CudaBufRef` is a shared view over a `CudaBuf`. `CudaBuf` is
// `Send + Sync` (its fields are `Vec<u8>`, `Mutex<_>`, and `AtomicBool`), so
// sharing and sending the view across threads is sound under the lifetime
// tie carried by `_marker`.
unsafe impl<'a> Sync for CudaBufRef<'a> {}
unsafe impl<'a> Send for CudaBufRef<'a> {}

impl<'a> Default for CudaBufRef<'a> {
    fn default() -> Self {
        Self {
            ptr: NonNull::dangling(),
            offset: 0,
            len: 0,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a> PartialEq for CudaBufRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr && self.offset == other.offset && self.len == other.len
    }
}

impl<'a> Eq for CudaBufRef<'a> {}

/// Mutable backend-native view into a CUDA buffer.
pub struct CudaBufMut<'a> {
    pub(crate) ptr: NonNull<CudaBuf>,
    pub(crate) offset: usize,
    pub(crate) len: usize,
    pub(crate) _marker: std::marker::PhantomData<&'a mut CudaBuf>,
}

// SAFETY: `CudaBufMut` is a uniquely borrowed mutable view tied to `'a`.
// `CudaBuf: Send + Sync` makes sending and sharing the view sound, matching
// the auto-impls for `&mut CudaBuf`.
unsafe impl<'a> Send for CudaBufMut<'a> {}
unsafe impl<'a> Sync for CudaBufMut<'a> {}

impl<'a> Default for CudaBufMut<'a> {
    fn default() -> Self {
        Self {
            ptr: NonNull::dangling(),
            offset: 0,
            len: 0,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a> PartialEq for CudaBufMut<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr && self.offset == other.offset && self.len == other.len
    }
}

impl<'a> Eq for CudaBufMut<'a> {}

impl fmt::Debug for CudaBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBuf")
            .field("len", &self.host.len())
            .field("device_allocated", &self.device.lock().unwrap().is_some())
            .field("device_stale", &self.device_stale.load(Ordering::Relaxed))
            .finish()
    }
}

impl PartialEq for CudaBuf {
    fn eq(&self, other: &Self) -> bool {
        self.host == other.host
    }
}

impl Eq for CudaBuf {}

impl CudaBuf {
    /// Allocates device memory and uploads the aligned host bytes.
    fn with_aligned_host(host: Vec<u8>) -> Self {
        let device = if host.is_empty() {
            None
        } else {
            Some(
                cuda_stream()
                    .clone_htod(host.as_slice())
                    .expect("failed to upload host bytes into CUDA device buffer"),
            )
        };

        Self {
            host,
            device: Mutex::new(device),
            device_stale: AtomicBool::new(false),
        }
    }

    fn aligned_clone(bytes: &[u8]) -> Vec<u8> {
        let mut host = alloc_aligned::<u8>(bytes.len());
        host.copy_from_slice(bytes);
        host
    }

    /// Pushes the host mirror to CUDA when a host-side mutation made it stale.
    fn ensure_device_current(&self) {
        if self.device_stale.swap(false, Ordering::AcqRel) && !self.host.is_empty() {
            let stream = cuda_stream();
            let mut device = self.device.lock().unwrap();
            stream
                .memcpy_htod(self.host.as_slice(), device.as_mut().expect("missing CUDA device buffer"))
                .expect("failed to copy host bytes to CUDA device buffer");
        }
    }

    /// Downloads the current CUDA buffer into a fresh aligned host vector.
    fn to_host_vec(&self) -> Vec<u8> {
        self.ensure_device_current();
        let mut host = alloc_aligned::<u8>(self.host.len());
        if !host.is_empty() {
            let device = self.device.lock().unwrap();
            cuda_stream()
                .memcpy_dtoh(device.as_ref().expect("missing CUDA device buffer"), &mut host)
                .expect("failed to copy CUDA device buffer to host");
        }
        host
    }
}

/// Lazily initialized CUDA context for device ordinal 0.
///
/// This keeps the prototype simple and makes the backend deterministic for the
/// current workstation setup. A production backend would likely make the
/// device/context configurable.
pub(crate) fn cuda_context() -> &'static Arc<CudaContext> {
    static CUDA_CONTEXT: OnceLock<Arc<CudaContext>> = OnceLock::new();
    CUDA_CONTEXT.get_or_init(|| CudaContext::new(0).expect("failed to initialize CUDA context for device 0"))
}

/// Returns the default stream associated with the shared CUDA context.
pub(crate) fn cuda_stream() -> Arc<CudaStream> {
    cuda_context().default_stream()
}

impl Backend for CudaGpuBackend {
    type ScalarPrep = f64;
    type ScalarBig = i64;
    type OwnedBuf = CudaBuf;
    type BufRef<'a> = CudaBufRef<'a>;
    type BufMut<'a> = CudaBufMut<'a>;
    type Handle = CudaFft64Handle;
    type Location = Device;

    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        CudaBuf::with_aligned_host(alloc_aligned::<u8>(len))
    }

    fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf {
        CudaBuf::with_aligned_host(CudaBuf::aligned_clone(bytes))
    }

    fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf {
        let host = if bytes.is_empty() || poulpy_hal::is_aligned(bytes.as_ptr()) {
            bytes
        } else {
            CudaBuf::aligned_clone(&bytes)
        };
        CudaBuf::with_aligned_host(host)
    }

    fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8> {
        buf.to_host_vec()
    }

    fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]) {
        assert_eq!(dst.len(), buf.host.len());
        buf.ensure_device_current();
        if !dst.is_empty() {
            let device = buf.device.lock().unwrap();
            cuda_stream()
                .memcpy_dtoh(device.as_ref().expect("missing CUDA device buffer"), dst)
                .expect("failed to copy CUDA device buffer to host");
        }
    }

    fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]) {
        assert_eq!(src.len(), buf.host.len());
        buf.host.copy_from_slice(src);
        buf.device_stale.store(false, Ordering::Release);
        if !src.is_empty() {
            let mut device = buf.device.lock().unwrap();
            cuda_stream()
                .memcpy_htod(src, device.as_mut().expect("missing CUDA device buffer"))
                .expect("failed to copy host bytes to CUDA device buffer");
        }
    }

    fn len_bytes(buf: &Self::OwnedBuf) -> usize {
        buf.host.len()
    }

    fn view(buf: &Self::OwnedBuf) -> Self::BufRef<'_> {
        CudaBufRef {
            ptr: NonNull::from(buf),
            offset: 0,
            len: buf.host.len(),
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
            len: buf.host.len(),
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

impl TransferFrom<poulpy_cpu_ref::FFT64Ref> for CudaGpuBackend {
    fn transfer_buf(src: &<poulpy_cpu_ref::FFT64Ref as Backend>::OwnedBuf) -> Self::OwnedBuf {
        Self::from_host_bytes(src)
    }
}

impl TransferFrom<CudaGpuBackend> for poulpy_cpu_ref::FFT64Ref {
    fn transfer_buf(src: &<CudaGpuBackend as Backend>::OwnedBuf) -> Self::OwnedBuf {
        let host = CudaGpuBackend::to_host_bytes(src);
        assert_alignment(host.as_ptr());
        host
    }
}

unsafe impl FFT64HandleFactory for CudaFft64Handle {
    fn create_fft64_handle(n: usize) -> Self {
        Self {
            table_fft: ReimFFTTable::new(n >> 1),
            table_ifft: ReimIFFTTable::new(n >> 1),
        }
    }

    fn assert_fft64_runtime_support() {
        let _ = cuda_context();
    }
}

unsafe impl FFTHandleProvider<f64> for CudaFft64Handle {
    fn get_fft_table(&self) -> &ReimFFTTable<f64> {
        &self.table_fft
    }

    fn get_ifft_table(&self) -> &ReimIFFTTable<f64> {
        &self.table_ifft
    }
}
