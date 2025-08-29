use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    reference::vec_znx::{
        test_vec_znx_add, test_vec_znx_add_inplace, test_vec_znx_sub, test_vec_znx_sub_ab_inplace, test_vec_znx_sub_ba_inplace,
    },
    tests::vec_znx::{test_vec_znx_add_normal, test_vec_znx_fill_uniform, test_vec_znx_normalize},
};

use crate::cpu_spqlios::FFT64;

#[test]
fn test_vec_znx_normalize_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_normalize(&module);
}

#[test]
fn test_vec_znx_add_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_add(&module);
}

#[test]
fn test_vec_znx_add_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_add_inplace(&module);
}

#[test]
fn test_vec_znx_sub_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_sub(&module);
}

#[test]
fn test_vec_znx_sub_ab_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_sub_ab_inplace(&module);
}

#[test]
fn test_vec_znx_sub_ba_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_sub_ba_inplace(&module);
}

#[test]
fn test_vec_znx_fill_uniform_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 12);
    test_vec_znx_fill_uniform(&module);
}

#[test]
fn test_vec_znx_add_normal_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 12);
    test_vec_znx_add_normal(&module);
}
