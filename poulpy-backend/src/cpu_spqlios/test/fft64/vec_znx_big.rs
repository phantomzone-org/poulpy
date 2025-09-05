use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    reference::vec_znx_big::{
        test_vec_znx_big_add, test_vec_znx_big_add_inplace, test_vec_znx_big_add_small, test_vec_znx_big_add_small_inplace,
    },
};

use crate::cpu_ref::FFT64;

#[test]
fn test_vec_znx_big_add_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_big_add(&module);
}

#[test]
fn test_vec_znx_big_add_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_big_add_inplace(&module);
}

#[test]
fn test_vec_znx_big_add_small_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_big_add_small(&module);
}

#[test]
fn test_vec_znx_big_add_small_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_big_add_small_inplace(&module);
}
