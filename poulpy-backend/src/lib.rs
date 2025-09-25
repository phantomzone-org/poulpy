#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod cpu_fft64_avx;

pub mod cpu_fft64_ref;
pub mod cpu_spqlios;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use cpu_fft64_avx::FFT64Avx;

pub use cpu_fft64_ref::FFT64Ref;
pub use cpu_spqlios::FFT64Spqlios;
