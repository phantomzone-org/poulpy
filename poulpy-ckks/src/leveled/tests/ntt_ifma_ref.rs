use std::sync::LazyLock;

use poulpy_cpu_ref::NTTIfmaRef;

use crate::leveled::tests::test_suite::{
    NTT120_PARAMS,
    add::{test_add, test_add_const, test_add_pt},
    conjugate::test_conjugate,
    encryption::test_encrypt_decrypt,
    helpers::TestContext,
    level::test_div_pow2,
    mul::{
        test_mul, test_mul_const, test_mul_int, test_mul_mismatched_delta, test_mul_mismatched_k, test_mul_pt,
        test_sequential_mul,
    },
    neg::test_neg,
    plaintext_prepared::{test_add_prepared_pt, test_mul_prepared_pt, test_sub_prepared_pt},
    rotate::test_rotate,
    sub::{test_sub, test_sub_const, test_sub_pt},
};

const ATK_ROTATIONS: &[i64] = &[1, 7];

// Reuse NTT120_PARAMS: same Q ~ 2^120, same base2k = 52.
static CTX: LazyLock<TestContext<NTTIfmaRef>> = LazyLock::new(|| TestContext::new(NTT120_PARAMS));
static CTX_TSK: LazyLock<TestContext<NTTIfmaRef>> = LazyLock::new(|| TestContext::new_with_tsk(NTT120_PARAMS));
static CTX_ATK: LazyLock<TestContext<NTTIfmaRef>> = LazyLock::new(|| TestContext::new_with_atk(NTT120_PARAMS, ATK_ROTATIONS));

#[test]
fn encrypt_decrypt() {
    test_encrypt_decrypt(&CTX);
}

#[test]
fn add() {
    test_add(&CTX);
}

#[test]
fn add_pt() {
    test_add_pt(&CTX);
}

#[test]
fn add_const() {
    test_add_const(&CTX);
}

#[test]
fn sub() {
    test_sub(&CTX);
}

#[test]
fn sub_pt() {
    test_sub_pt(&CTX);
}

#[test]
fn sub_const() {
    test_sub_const(&CTX);
}

#[test]
fn neg() {
    test_neg(&CTX);
}

#[test]
fn div_pow2() {
    test_div_pow2(&CTX);
}

#[test]
fn mul_int() {
    test_mul_int(&CTX);
}

#[test]
fn mul_pt() {
    test_mul_pt(&CTX);
}

#[test]
fn mul_const() {
    test_mul_const(&CTX);
}

#[test]
fn mul() {
    test_mul(&CTX_TSK);
}

#[test]
fn mul_mismatched_k() {
    test_mul_mismatched_k(&CTX_TSK);
}

#[test]
fn mul_mismatched_delta() {
    test_mul_mismatched_delta(&CTX_TSK);
}

#[test]
fn sequential_mul() {
    test_sequential_mul(&CTX_TSK, 2);
}

#[test]
fn add_prepared_pt() {
    test_add_prepared_pt(&CTX);
}

#[test]
fn sub_prepared_pt() {
    test_sub_prepared_pt(&CTX);
}

#[test]
fn mul_prepared_pt() {
    test_mul_prepared_pt(&CTX);
}

#[test]
fn rotate() {
    test_rotate(&CTX_ATK, ATK_ROTATIONS);
}

#[test]
fn conjugate() {
    test_conjugate(&CTX_ATK);
}
