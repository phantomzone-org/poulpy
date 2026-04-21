mod ckks_impl;

pub use ckks_impl::CKKSImpl;

unsafe impl CKKSImpl<poulpy_cpu_ref::FFT64Ref> for poulpy_cpu_ref::FFT64Ref {
    crate::impl_ckks_default_methods!(poulpy_cpu_ref::FFT64Ref);
}

unsafe impl CKKSImpl<poulpy_cpu_ref::NTT120Ref> for poulpy_cpu_ref::NTT120Ref {
    crate::impl_ckks_default_methods!(poulpy_cpu_ref::NTT120Ref);
}

unsafe impl CKKSImpl<poulpy_cpu_ref::NTTIfmaRef> for poulpy_cpu_ref::NTTIfmaRef {
    crate::impl_ckks_default_methods!(poulpy_cpu_ref::NTTIfmaRef);
}

#[cfg(feature = "enable-avx")]
unsafe impl CKKSImpl<poulpy_cpu_avx::FFT64Avx> for poulpy_cpu_avx::FFT64Avx {
    crate::impl_ckks_default_methods!(poulpy_cpu_avx::FFT64Avx);
}

#[cfg(feature = "enable-avx")]
unsafe impl CKKSImpl<poulpy_cpu_avx::NTT120Avx> for poulpy_cpu_avx::NTT120Avx {
    crate::impl_ckks_default_methods!(poulpy_cpu_avx::NTT120Avx);
}

#[cfg(feature = "enable-ifma")]
unsafe impl CKKSImpl<poulpy_cpu_ifma::NTTIfma> for poulpy_cpu_ifma::NTTIfma {
    crate::impl_ckks_default_methods!(poulpy_cpu_ifma::NTTIfma);
}
