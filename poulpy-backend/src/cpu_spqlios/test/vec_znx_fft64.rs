use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    tests::vec_znx::{
        test_vec_znx_add, test_vec_znx_add_normal, test_vec_znx_fill_uniform, test_vec_znx_negate, test_vec_znx_normalize,
        test_vec_znx_sub,
    },
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
fn test_vec_znx_sub_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_sub(&module);
}

#[test]
fn test_vec_znx_negate_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_negate(&module);
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
