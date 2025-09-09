use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    reference::vec_znx_dft::fft64::{
        test_vec_znx_dft_add, test_vec_znx_dft_add_inplace, test_vec_znx_dft_sub, test_vec_znx_dft_sub_ab_inplace,
        test_vec_znx_dft_sub_ba_inplace,
    },
};

use crate::cpu_spqlios::FFT64;

#[test]
fn test_vec_znx_dft_add_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_dft_add(&module);
}

#[test]
fn test_vec_znx_dft_add_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_dft_add_inplace(&module);
}

#[test]
fn test_vec_znx_dft_sub_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_dft_sub(&module);
}

#[test]
fn test_vec_znx_dft_sub_ab_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_dft_sub_ab_inplace(&module);
}

#[test]
fn test_vec_znx_dft_sub_ba_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_dft_sub_ba_inplace(&module);
}
