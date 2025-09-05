use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    reference::vec_znx_big::{
        test_vec_znx_big_add, test_vec_znx_big_add_inplace, test_vec_znx_big_add_normal, test_vec_znx_big_add_small,
        test_vec_znx_big_add_small_inplace, test_vec_znx_big_negate, test_vec_znx_big_negate_inplace, test_vec_znx_big_normalize,
        test_vec_znx_big_sub, test_vec_znx_big_sub_ab_inplace, test_vec_znx_big_sub_ba_inplace, test_vec_znx_big_sub_small_a,
        test_vec_znx_big_sub_small_a_inplace, test_vec_znx_big_sub_small_b, test_vec_znx_big_sub_small_b_inplace,
    },
};

use crate::cpu_spqlios::FFT64;

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

#[test]
fn test_vec_znx_big_sub_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_big_sub(&module);
}

#[test]
fn test_vec_znx_big_sub_ab_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_big_sub_ab_inplace(&module);
}

#[test]
fn test_vec_znx_big_sub_ba_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_big_sub_ba_inplace(&module);
}

#[test]
fn test_vec_znx_big_sub_small_a_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_big_sub_small_a(&module);
}

#[test]
fn test_vec_znx_big_sub_small_b_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_big_sub_small_b(&module);
}

#[test]
fn test_vec_znx_big_sub_small_a_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_big_sub_small_a_inplace(&module);
}

#[test]
fn test_vec_znx_big_sub_small_b_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_big_sub_small_b_inplace(&module);
}

#[test]
fn test_vec_znx_big_add_normal_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 12);
    test_vec_znx_big_add_normal(&module);
}

#[test]
fn test_vec_znx_big_negate_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_big_negate(&module);
}

#[test]
fn test_vec_znx_big_negate_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_big_negate_inplace(&module);
}

#[test]
fn test_vec_znx_big_normalize_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_big_normalize(&module);
}
