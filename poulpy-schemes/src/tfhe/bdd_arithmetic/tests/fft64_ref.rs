use std::sync::LazyLock;

use poulpy_backend::FFT64Ref;

use crate::tfhe::{
    bdd_arithmetic::tests::test_suite::{
        TestContext, test_bdd_add, test_bdd_and, test_bdd_or, test_bdd_prepare, test_bdd_sll, test_bdd_slt, test_bdd_sltu,
        test_bdd_sra, test_bdd_srl, test_bdd_sub, test_bdd_xor, test_fhe_uint_splice_u8, test_fhe_uint_splice_u16,
        test_glwe_blind_selection, test_glwe_to_glwe_blind_rotation, test_scalar_to_ggsw_blind_rotation,
    },
    blind_rotation::CGGI,
};

static TEST_CONTEXT_CGGI_FFT64_REF: LazyLock<TestContext<CGGI, FFT64Ref>> =
    LazyLock::new(|| TestContext::<CGGI, FFT64Ref>::new());

#[test]
fn test_glwe_blind_selection_fft64_ref() {
    test_glwe_blind_selection(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_fhe_uint_splice_u8_fft64_ref() {
    test_fhe_uint_splice_u8(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_fhe_uint_splice_u16_fft64_ref() {
    test_fhe_uint_splice_u16(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_glwe_to_glwe_blind_rotation_fft64_ref() {
    test_glwe_to_glwe_blind_rotation(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_scalar_to_ggsw_blind_rotation_fft64_ref() {
    test_scalar_to_ggsw_blind_rotation(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_bdd_prepare_fft64_ref() {
    test_bdd_prepare(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_bdd_add_fft64_ref() {
    test_bdd_add(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_bdd_and_fft64_ref() {
    test_bdd_and(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_bdd_or_fft64_ref() {
    test_bdd_or(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_bdd_sll_fft64_ref() {
    test_bdd_sll(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_bdd_slt_fft64_ref() {
    test_bdd_slt(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_bdd_sltu_fft64_ref() {
    test_bdd_sltu(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_bdd_sra_fft64_ref() {
    test_bdd_sra(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_bdd_srl_fft64_ref() {
    test_bdd_srl(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_bdd_sub_fft64_ref() {
    test_bdd_sub(&TEST_CONTEXT_CGGI_FFT64_REF)
}

#[test]
fn test_bdd_xor_fft64_ref() {
    test_bdd_xor(&TEST_CONTEXT_CGGI_FFT64_REF)
}
