use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    reference::fft64::vec_znx_big::test_vec_znx_big_add_normal,
    test_suite::vec_znx_big::{
        test_vec_znx_big_add, test_vec_znx_big_add_inplace, test_vec_znx_big_add_small, test_vec_znx_big_add_small_inplace,
        test_vec_znx_big_automorphism, test_vec_znx_big_automorphism_inplace, test_vec_znx_big_negate,
        test_vec_znx_big_negate_inplace, test_vec_znx_big_normalize, test_vec_znx_big_sub, test_vec_znx_big_sub_ab_inplace,
        test_vec_znx_big_sub_ba_inplace, test_vec_znx_big_sub_small_a, test_vec_znx_big_sub_small_a_inplace,
        test_vec_znx_big_sub_small_b, test_vec_znx_big_sub_small_b_inplace,
    },
};

use crate::{FFT64Avx, FFT64Ref};

#[test]
fn test_vec_znx_big_add_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_add(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_add_inplace_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_add_inplace(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_add_small_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_add_small(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_add_small_inplace_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_add_small_inplace(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_sub_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_sub(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_sub_ab_inplace_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_sub_ab_inplace(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_sub_ba_inplace_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_sub_ba_inplace(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_sub_small_a_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_sub_small_a(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_sub_small_b_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_sub_small_b(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_sub_small_a_inplace_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_sub_small_a_inplace(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_sub_small_b_inplace_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_sub_small_b_inplace(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_add_normal_fft64() {
    let module: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 12);
    test_vec_znx_big_add_normal(&module);
}

#[test]
fn test_vec_znx_big_negate_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_negate(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_negate_inplace_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_negate_inplace(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_automorphism_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_automorphism(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_automorphism_inplace_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_automorphism_inplace(12, &module_ref, &module_test)
}

#[test]
fn test_vec_znx_big_normalize_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_big_normalize(12, &module_ref, &module_test)
}
