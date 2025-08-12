mod ffi;
mod mat_znx;
mod module_fft64;
mod module_ntt120;
mod scalar_znx;
mod scratch;
mod svp_ppol_fft64;
mod svp_ppol_ntt120;
mod vec_znx;
mod vec_znx_big_fft64;
mod vec_znx_big_ntt120;
mod vec_znx_dft_fft64;
mod vec_znx_dft_ntt120;
mod vmp_pmat_fft64;
mod vmp_pmat_ntt120;

#[cfg(test)]
mod test;

pub use module_fft64::*;
pub use module_ntt120::*;

/// For external documentation
pub use vec_znx::{
    vec_znx_copy_ref, vec_znx_lsh_inplace_ref, vec_znx_merge_ref, vec_znx_rsh_inplace_ref, vec_znx_split_ref,
    vec_znx_switch_degree_ref,
};

pub trait CPUAVX {}
