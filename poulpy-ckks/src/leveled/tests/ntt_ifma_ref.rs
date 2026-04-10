use std::sync::LazyLock;

use poulpy_cpu_ref::NTTIfmaRef;

use crate::leveled::tests::test_suite::{NTT120_PARAMS, helpers::TestContext};

const ATK_ROTATIONS: &[i64] = &[1, 7];

// Reuse NTT120_PARAMS: same Q ~ 2^120, same base2k = 52.
static CTX: LazyLock<TestContext<NTTIfmaRef>> = LazyLock::new(|| TestContext::new(NTT120_PARAMS, ATK_ROTATIONS));

#[test]
fn encrypt_decrypt() {
    crate::leveled::tests::test_suite::encryption::test_encrypt_decrypt(&CTX);
}

#[test]
fn add_ct_aligned() {
    crate::leveled::tests::test_suite::add::test_add_ct_aligned(&CTX);
}

#[test]
fn add_ct_delta_a_lt_b() {
    crate::leveled::tests::test_suite::add::test_add_ct_delta_a_lt_b(&CTX);
}

#[test]
fn add_ct_delta_a_gt_b() {
    crate::leveled::tests::test_suite::add::test_add_ct_delta_a_gt_b(&CTX);
}

#[test]
fn add_ct_smaller_output() {
    crate::leveled::tests::test_suite::add::test_add_ct_smaller_output(&CTX);
}

#[test]
fn add_ct_inplace_aligned() {
    crate::leveled::tests::test_suite::add::test_add_ct_inplace_aligned(&CTX);
}

#[test]
fn add_ct_inplace_self_lt() {
    crate::leveled::tests::test_suite::add::test_add_ct_inplace_self_lt(&CTX);
}

#[test]
fn add_ct_inplace_self_gt() {
    crate::leveled::tests::test_suite::add::test_add_ct_inplace_self_gt(&CTX);
}

#[test]
fn add_pt_znx_inplace() {
    crate::leveled::tests::test_suite::add::test_add_pt_znx_inplace(&CTX);
}

#[test]
fn add_pt_znx() {
    crate::leveled::tests::test_suite::add::test_add_pt_znx(&CTX);
}

#[test]
fn add_pt_rnx_inplace() {
    crate::leveled::tests::test_suite::add::test_add_pt_rnx_inplace(&CTX);
}

#[test]
fn add_pt_rnx() {
    crate::leveled::tests::test_suite::add::test_add_pt_rnx(&CTX);
}

#[test]
fn add_pt_znx_smaller_output() {
    crate::leveled::tests::test_suite::add::test_add_pt_znx_smaller_output(&CTX);
}

#[test]
fn add_pt_rnx_smaller_output() {
    crate::leveled::tests::test_suite::add::test_add_pt_rnx_smaller_output(&CTX);
}

#[test]
fn sub_ct_aligned() {
    crate::leveled::tests::test_suite::sub::test_sub_ct_aligned(&CTX);
}

#[test]
fn sub_ct_delta_a_lt_b() {
    crate::leveled::tests::test_suite::sub::test_sub_ct_delta_a_lt_b(&CTX);
}

#[test]
fn sub_ct_delta_a_gt_b() {
    crate::leveled::tests::test_suite::sub::test_sub_ct_delta_a_gt_b(&CTX);
}

#[test]
fn sub_ct_smaller_output() {
    crate::leveled::tests::test_suite::sub::test_sub_ct_smaller_output(&CTX);
}

#[test]
fn sub_ct_inplace_aligned() {
    crate::leveled::tests::test_suite::sub::test_sub_ct_inplace_aligned(&CTX);
}

#[test]
fn sub_ct_inplace_self_lt() {
    crate::leveled::tests::test_suite::sub::test_sub_ct_inplace_self_lt(&CTX);
}

#[test]
fn sub_ct_inplace_self_gt() {
    crate::leveled::tests::test_suite::sub::test_sub_ct_inplace_self_gt(&CTX);
}

#[test]
fn sub_pt_znx_inplace() {
    crate::leveled::tests::test_suite::sub::test_sub_pt_znx_inplace(&CTX);
}

#[test]
fn sub_pt_znx() {
    crate::leveled::tests::test_suite::sub::test_sub_pt_znx(&CTX);
}

#[test]
fn sub_pt_rnx_inplace() {
    crate::leveled::tests::test_suite::sub::test_sub_pt_rnx_inplace(&CTX);
}

#[test]
fn sub_pt_rnx() {
    crate::leveled::tests::test_suite::sub::test_sub_pt_rnx(&CTX);
}

#[test]
fn sub_pt_znx_smaller_output() {
    crate::leveled::tests::test_suite::sub::test_sub_pt_znx_smaller_output(&CTX);
}

#[test]
fn sub_pt_rnx_smaller_output() {
    crate::leveled::tests::test_suite::sub::test_sub_pt_rnx_smaller_output(&CTX);
}

#[test]
fn neg() {
    crate::leveled::tests::test_suite::neg::test_neg(&CTX);
}

#[test]
fn neg_inplace() {
    crate::leveled::tests::test_suite::neg::test_neg_inplace(&CTX);
}

#[test]
fn conjugate() {
    crate::leveled::tests::test_suite::conjugate::test_conjugate(&CTX);
}

#[test]
fn conjugate_inplace() {
    crate::leveled::tests::test_suite::conjugate::test_conjugate_inplace(&CTX);
}

#[test]
fn rotate() {
    crate::leveled::tests::test_suite::rotate::test_rotate(&CTX, ATK_ROTATIONS);
}

#[test]
fn rotate_inplace() {
    crate::leveled::tests::test_suite::rotate::test_rotate_inplace(&CTX, ATK_ROTATIONS);
}

#[test]
fn mul_ct_aligned() {
    crate::leveled::tests::test_suite::mul::test_mul_ct_aligned(&CTX);
}
