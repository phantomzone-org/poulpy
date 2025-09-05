use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    reference::vec_znx::{
        test_vec_znx_add, test_vec_znx_add_inplace, test_vec_znx_add_scalar, test_vec_znx_add_scalar_inplace,
        test_vec_znx_automorphism, test_vec_znx_automorphism_inplace, test_vec_znx_lsh, test_vec_znx_lsh_inplace,
        test_vec_znx_merge_rings, test_vec_znx_mul_xp_minus_one, test_vec_znx_mul_xp_minus_one_inplace, test_vec_znx_negate,
        test_vec_znx_negate_inplace, test_vec_znx_normalize, test_vec_znx_normalize_inplace, test_vec_znx_rotate,
        test_vec_znx_rotate_inplace, test_vec_znx_rsh, test_vec_znx_rsh_inplace, test_vec_znx_split_ring, test_vec_znx_sub,
        test_vec_znx_sub_ab_inplace, test_vec_znx_sub_ba_inplace, test_vec_znx_sub_scalar, test_vec_znx_sub_scalar_inplace,
        test_vec_znx_switch_ring,
    },
    tests::vec_znx::{test_vec_znx_add_normal, test_vec_znx_fill_uniform},
};

use crate::cpu_spqlios::FFT64;

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
fn test_vec_znx_add_scalar_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_add_scalar(&module);
}

#[test]
fn test_vec_znx_add_scalar_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_add_scalar_inplace(&module);
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
fn test_vec_znx_sub_scalar_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_sub_scalar(&module);
}

#[test]
fn test_vec_znx_sub_scalar_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_sub_scalar_inplace(&module);
}

#[test]
fn test_vec_znx_negate_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_negate(&module);
}

#[test]
fn test_vec_znx_negate_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_negate_inplace(&module);
}

#[test]
fn test_vec_znx_rsh_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_rsh(&module);
}

#[test]
fn test_vec_znx_rsh_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_rsh_inplace(&module);
}

#[test]
fn test_vec_znx_lsh_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_lsh(&module);
}

#[test]
fn test_vec_znx_lsh_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_lsh_inplace(&module);
}

#[test]
fn test_vec_znx_rotate_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_rotate(&module);
}

#[test]
fn test_vec_znx_rotate_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_rotate_inplace(&module);
}

#[test]
fn test_vec_znx_automorphism_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_automorphism(&module);
}

#[test]
fn test_vec_znx_automorphism_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_automorphism_inplace(&module);
}

#[test]
fn test_vec_znx_mul_xp_minus_one_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_mul_xp_minus_one(&module);
}

#[test]
fn test_vec_znx_mul_xp_minus_one_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_mul_xp_minus_one_inplace(&module);
}

#[test]
fn test_vec_znx_normalize_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_normalize(&module);
}

#[test]
fn test_vec_znx_normalize_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_normalize_inplace(&module);
}

#[test]
fn test_vec_znx_switch_ring_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_switch_ring(&module);
}

#[test]
fn test_vec_znx_split_ring_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_split_ring(&module);
}

#[test]
fn test_vec_znx_merge_rings_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_merge_rings(&module);
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
