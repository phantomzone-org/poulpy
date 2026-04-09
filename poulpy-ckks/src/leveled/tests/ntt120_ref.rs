use std::sync::LazyLock;

use poulpy_cpu_ref::NTT120Ref;

use crate::leveled::tests::test_suite::{NTT120_PARAMS, helpers::TestContext};

const ATK_ROTATIONS: &[i64] = &[1, 7];

static CTX: LazyLock<TestContext<NTT120Ref>> = LazyLock::new(|| TestContext::new(NTT120_PARAMS, ATK_ROTATIONS));

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
fn mul_ct_aligned() {
    crate::leveled::tests::test_suite::mul::test_mul_ct_aligned(&CTX);
}

/*
#[test]

#[test]
fn neg() {
    test_neg(&CTX);
}

#[test]
fn div_pow2() {
    test_div_pow2(&CTX);
}

#[test]
fn drop_scaling_precision() {
    test_drop_scaling_precision(&CTX);
}

#[test]
fn drop_torus_precision() {
    test_drop_torus_precision(&CTX);
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
fn mul_aligned() {
    test_mul_aligned(&CTX_TSK);
}

#[test]
fn square() {
    test_square(&CTX_TSK);
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
fn deep_square_chain() {
    test_deep_square_chain(&CTX_TSK);
}

#[test]
fn square_size_reduced_input() {
    test_square_size_reduced_input(&CTX_TSK);
}

#[test]
fn mul_size_reduced_inputs() {
    test_mul_size_reduced_inputs(&CTX_TSK);
}

#[test]
fn mul_tmp_bytes_scales_with_size() {
    test_mul_tmp_bytes_scales_with_size(&CTX_TSK);
}

#[test]
fn add_prepared_pt() {
    test_add_prepared_pt(&CTX);
}

#[test]
fn prepared_linear_sum() {
    test_prepared_linear_sum(&CTX);
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
fn prepared_poly2_sum() {
    test_prepared_poly2_sum(&CTX_TSK);
}

#[test]
fn prepared_poly2_sum_aligned() {
    test_prepared_poly2_sum_aligned(&CTX_TSK);
}

#[test]
fn prepared_poly2_term_align() {
    test_prepared_poly2_term_align(&CTX_TSK);
}

#[test]
fn prepared_poly2_term() {
    test_prepared_poly2_term(&CTX_TSK);
}

#[test]
fn prepared_poly2_mul() {
    test_prepared_poly2_mul(&CTX_TSK);
}

#[test]
fn rotate() {
    test_rotate(&CTX_ATK, ATK_ROTATIONS);
}

#[test]
fn conjugate() {
    test_conjugate(&CTX_ATK);
}
    */
