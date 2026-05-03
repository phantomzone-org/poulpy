//! Narrow HAL-family implementations owned by `poulpy-gpu-cuda`.
//!
//! This file is intentionally small. It documents the recommended pattern for
//! an external device backend:
//! - keep the impl surface narrow
//! - avoid inheriting host-only CPU defaults when the backend is device-only

use std::ptr::NonNull;

use crate::CudaGpuBackend;
use poulpy_cpu_ref::reference::fft64::module::FFT64HandleFactory;
use poulpy_hal::{layouts::Module, oep::HalModuleImpl};

unsafe impl HalModuleImpl<CudaGpuBackend> for CudaGpuBackend {
    fn new(n: u64) -> Module<CudaGpuBackend> {
        <crate::CudaFft64Handle as FFT64HandleFactory>::assert_fft64_runtime_support();
        let handle = <crate::CudaFft64Handle as FFT64HandleFactory>::create_fft64_handle(n as usize);
        let ptr: NonNull<crate::CudaFft64Handle> = NonNull::from(Box::leak(Box::new(handle)));
        unsafe { Module::from_nonnull(ptr, n) }
    }
}
