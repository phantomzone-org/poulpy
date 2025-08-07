pub(crate) mod ffi;
pub(crate) mod mat_znx;
pub(crate) mod module_fft64;
pub(crate) mod module_ntt120;
pub(crate) mod scalar_znx;
pub(crate) mod scratch;
pub(crate) mod svp_ppol_fft64;
pub(crate) mod svp_ppol_ntt120;
pub(crate) mod vec_znx;
pub(crate) mod vec_znx_big_fft64;
pub(crate) mod vec_znx_big_ntt120;
pub(crate) mod vec_znx_dft_fft64;
pub(crate) mod vec_znx_dft_ntt120;
pub(crate) mod vmp_pmat_fft64;
pub(crate) mod vmp_pmat_ntt120;

#[cfg(test)]
mod test;

pub use module_fft64::*;
pub use module_ntt120::*;

pub(crate) trait CPUAVX {}
