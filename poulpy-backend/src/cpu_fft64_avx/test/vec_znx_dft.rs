use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    test_suite::vec_znx_dft::{
        test_vec_znx_dft_add, test_vec_znx_dft_add_inplace, test_vec_znx_dft_sub, test_vec_znx_dft_sub_ab_inplace,
        test_vec_znx_dft_sub_ba_inplace, test_vec_znx_idft_apply, test_vec_znx_idft_apply_consume, test_vec_znx_idft_apply_tmpa,
    },
};

use crate::{FFT64Avx, FFT64Ref};

#[test]
fn test_vec_znx_dft_add_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_dft_add(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_dft_add_inplace_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_dft_add_inplace(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_dft_sub_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_dft_sub(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_dft_sub_ab_inplace_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_dft_sub_ab_inplace(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_dft_sub_ba_inplace_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_dft_sub_ba_inplace(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_idft_apply_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_idft_apply(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_idft_apply_tmpa_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_idft_apply_tmpa(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_idft_apply_consume_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_idft_apply_consume(12, &module_ref, &module_test)
}

// #[test]
// fn test_vec_znx_copy_fft64() {
// let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
// let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
// test_vec_znx_dft_copy(12, &module_ref, &module_test)
// }
