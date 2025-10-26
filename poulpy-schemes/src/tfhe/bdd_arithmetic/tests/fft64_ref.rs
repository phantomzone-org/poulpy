use poulpy_backend::FFT64Ref;

use crate::tfhe::{
    bdd_arithmetic::tests::test_suite::{
        test_bdd_add, test_bdd_and, test_bdd_or, test_bdd_prepare, test_bdd_sll, test_bdd_slt, test_bdd_sltu, test_bdd_sra,
        test_bdd_srl, test_bdd_sub, test_bdd_xor, test_glwe_to_glwe_blind_rotation, test_scalar_to_ggsw_blind_rotation,
    },
    blind_rotation::CGGI,
};

#[test]
fn test_glwe_to_glwe_blind_rotation_fft64_ref() {
    test_glwe_to_glwe_blind_rotation::<FFT64Ref>()
}

#[test]
fn test_scalar_to_ggsw_blind_rotation_fft64_ref() {
    test_scalar_to_ggsw_blind_rotation::<FFT64Ref>()
}

#[test]
fn test_bdd_prepare_fft64_ref() {
    test_bdd_prepare::<CGGI, FFT64Ref>()
}

#[test]
fn test_bdd_add_fft64_ref() {
    test_bdd_add::<CGGI, FFT64Ref>()
}

#[test]
fn test_bdd_and_fft64_ref() {
    test_bdd_and::<CGGI, FFT64Ref>()
}

#[test]
fn test_bdd_or_fft64_ref() {
    test_bdd_or::<CGGI, FFT64Ref>()
}

#[test]
fn test_bdd_sll_fft64_ref() {
    test_bdd_sll::<CGGI, FFT64Ref>()
}

#[test]
fn test_bdd_slt_fft64_ref() {
    test_bdd_slt::<CGGI, FFT64Ref>()
}

#[test]
fn test_bdd_sltu_fft64_ref() {
    test_bdd_sltu::<CGGI, FFT64Ref>()
}

#[test]
fn test_bdd_sra_fft64_ref() {
    test_bdd_sra::<CGGI, FFT64Ref>()
}

#[test]
fn test_bdd_srl_fft64_ref() {
    test_bdd_srl::<CGGI, FFT64Ref>()
}

#[test]
fn test_bdd_sub_fft64_ref() {
    test_bdd_sub::<CGGI, FFT64Ref>()
}

#[test]
fn test_bdd_xor_fft64_ref() {
    test_bdd_xor::<CGGI, FFT64Ref>()
}
