#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

mod module;
mod reim;
mod reim4;
mod scratch;
mod svp;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_dft;
mod vmp;
mod znx_avx;

pub struct FFT64Avx {}
pub use reim::*;

#[cfg(test)]
pub mod tests;
