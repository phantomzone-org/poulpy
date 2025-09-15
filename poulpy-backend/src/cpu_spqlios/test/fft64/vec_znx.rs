use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    test_suite::vec_znx::{
        test_vec_znx_add, test_vec_znx_add_inplace, test_vec_znx_add_normal, test_vec_znx_add_scalar,
        test_vec_znx_add_scalar_inplace, test_vec_znx_automorphism, test_vec_znx_automorphism_inplace, test_vec_znx_copy,
        test_vec_znx_fill_normal, test_vec_znx_fill_uniform, test_vec_znx_lsh, test_vec_znx_lsh_inplace,
        test_vec_znx_merge_rings, test_vec_znx_mul_xp_minus_one, test_vec_znx_mul_xp_minus_one_inplace, test_vec_znx_negate,
        test_vec_znx_negate_inplace, test_vec_znx_normalize, test_vec_znx_normalize_inplace, test_vec_znx_rotate,
        test_vec_znx_rotate_inplace, test_vec_znx_rsh, test_vec_znx_rsh_inplace, test_vec_znx_split_ring, test_vec_znx_sub,
        test_vec_znx_sub_ab_inplace, test_vec_znx_sub_ba_inplace, test_vec_znx_sub_scalar, test_vec_znx_sub_scalar_inplace,
        test_vec_znx_switch_ring,
    },
};

use crate::{cpu_fft64_ref::FFT64Ref, cpu_spqlios::FFT64Spqlios};

#[test]
fn test_vec_znx_add_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_add(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_add_inplace_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_add_inplace(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_add_scalar_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_add_scalar(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_add_scalar_inplace_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_add_scalar_inplace(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_sub_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_sub(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_sub_ab_inplace_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_sub_ab_inplace(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_sub_ba_inplace_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_sub_ba_inplace(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_sub_scalar_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_sub_scalar(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_sub_scalar_inplace_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_sub_scalar_inplace(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_negate_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_negate(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_negate_inplace_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_negate_inplace(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_rsh_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_rsh(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_rsh_inplace_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_rsh_inplace(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_lsh_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_lsh(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_lsh_inplace_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_lsh_inplace(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_rotate_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_rotate(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_rotate_inplace_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_rotate_inplace(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_automorphism_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_automorphism(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_automorphism_inplace_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_automorphism_inplace(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_mul_xp_minus_one_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_mul_xp_minus_one(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_mul_xp_minus_one_inplace_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_mul_xp_minus_one_inplace(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_normalize_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_normalize(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_normalize_inplace_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_normalize_inplace(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_switch_ring_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_switch_ring(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_split_ring_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_split_ring(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_merge_rings_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_merge_rings(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_copy_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vec_znx_copy(12, &module_ref, &module_test);
}

#[test]
fn test_vec_znx_fill_uniform_fft64() {
    let module: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 12);
    test_vec_znx_fill_uniform(&module);
}

#[test]
fn test_vec_znx_fill_normal_fft64() {
    let module: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 12);
    test_vec_znx_fill_normal(&module);
}

#[test]
fn test_vec_znx_add_normal_fft64() {
    let module: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 12);
    test_vec_znx_add_normal(&module);
}
