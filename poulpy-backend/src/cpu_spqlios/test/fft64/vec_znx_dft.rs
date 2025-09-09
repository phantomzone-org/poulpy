use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    reference::{
        vec_znx::test_vec_znx_copy,
        vec_znx_dft::fft64::{
            test_vec_znx_dft_add, test_vec_znx_dft_add_inplace, test_vec_znx_dft_apply, test_vec_znx_dft_sub,
            test_vec_znx_dft_sub_ab_inplace, test_vec_znx_dft_sub_ba_inplace, test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume, test_vec_znx_idft_apply_tmpa,
        },
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

#[test]
fn test_vec_znx_dft_apply_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_dft_apply(&module);
}

#[test]
fn test_vec_znx_idft_apply_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_idft_apply(&module);
}

#[test]
fn test_vec_znx_idft_apply_tmpa_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_idft_apply_tmpa(&module);
}

#[test]
fn test_vec_znx_idft_apply_consume_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_idft_apply_consume(&module);
}

#[test]
fn test_vec_znx_copy_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_copy(&module);
}
