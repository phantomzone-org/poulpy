use std::sync::LazyLock;

use poulpy_cpu_ref::NTT120Ref;

use crate::leveled::tests::test_suite::{NTT120_PARAMS, helpers::TestContext};

const ATK_ROTATIONS: &[i64] = &[1, 7];

static CTX: LazyLock<TestContext<NTT120Ref>> = LazyLock::new(|| TestContext::new(NTT120_PARAMS, ATK_ROTATIONS));

use anyhow::Result;

#[test]
fn encrypt_decrypt() {
    crate::leveled::tests::test_suite::encryption::test_encrypt_decrypt(&CTX);
}

#[test]
fn set_log_decimal_out_of_range_error() {
    crate::leveled::tests::test_suite::errors::test_set_log_decimal_out_of_range_error(&CTX);
}

#[test]
fn set_log_hom_rem_out_of_range_error() {
    crate::leveled::tests::test_suite::errors::test_set_log_hom_rem_out_of_range_error(&CTX);
}

#[test]
fn reallocate_limbs_checked_error() {
    crate::leveled::tests::test_suite::errors::test_reallocate_limbs_checked_error(&CTX);
}

#[test]
fn add_pt_znx_alignment_error() {
    crate::leveled::tests::test_suite::errors::test_add_pt_znx_alignment_error(&CTX);
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
fn add_ct_delta_log_decimal() {
    crate::leveled::tests::test_suite::add::test_add_ct_delta_log_decimal(&CTX);
}

#[test]
fn add_ct_aligned_smaller_output() {
    crate::leveled::tests::test_suite::add::test_add_ct_aligned_smaller_output(&CTX);
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
fn add_pt_znx_aligned() {
    crate::leveled::tests::test_suite::add::test_add_pt_znx_aligned(&CTX);
}

#[test]
fn add_pt_znx_delta_log_decimal() {
    crate::leveled::tests::test_suite::add::test_add_pt_znx_delta_log_decimal(&CTX);
}

#[test]
fn add_pt_rnx_inplace() {
    crate::leveled::tests::test_suite::add::test_add_pt_rnx_inplace(&CTX);
}

#[test]
fn add_pt_rnx_aligned() {
    crate::leveled::tests::test_suite::add::test_add_pt_rnx_aligned(&CTX);
}

#[test]
fn add_pt_rnx_delta_log_decimal() {
    crate::leveled::tests::test_suite::add::test_add_pt_rnx_delta_log_decimal(&CTX);
}

#[test]
fn add_pt_znx_smaller_output() {
    crate::leveled::tests::test_suite::add::test_add_pt_znx_smaller_output(&CTX);
}

#[test]
fn add_pt_znx_base2k_mismatch_error() {
    crate::leveled::tests::test_suite::add::test_add_pt_znx_base2k_mismatch_error(&CTX);
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
fn sub_ct_delta_log_decimal() {
    crate::leveled::tests::test_suite::sub::test_sub_ct_delta_log_decimal(&CTX);
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
fn sub_pt_znx_delta_log_decimal() {
    crate::leveled::tests::test_suite::sub::test_sub_pt_znx_delta_log_decimal(&CTX);
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
fn sub_pt_rnx_delta_log_decimal() {
    crate::leveled::tests::test_suite::sub::test_sub_pt_rnx_delta_log_decimal(&CTX);
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
fn neg() -> Result<()> {
    crate::leveled::tests::test_suite::neg::test_neg_aligned(&CTX)
}

#[test]
fn neg_smaller_output() -> Result<()> {
    crate::leveled::tests::test_suite::neg::test_neg_smaller_output(&CTX)
}

#[test]
fn neg_inplace() {
    crate::leveled::tests::test_suite::neg::test_neg_inplace(&CTX);
}

#[test]
fn conjugate_aligned() {
    crate::leveled::tests::test_suite::conjugate::test_conjugate_aligned(&CTX);
}

#[test]
fn conjugate_smaller_output() {
    crate::leveled::tests::test_suite::conjugate::test_conjugate_smaller_output(&CTX);
}

#[test]
fn conjugate_inplace() {
    crate::leveled::tests::test_suite::conjugate::test_conjugate_inplace(&CTX);
}

#[test]
fn rotate_aligned() {
    crate::leveled::tests::test_suite::rotate::test_rotate_aligned(&CTX, ATK_ROTATIONS);
}

#[test]
fn rotate_smaller_output() {
    crate::leveled::tests::test_suite::rotate::test_rotate_smaller_output(&CTX, ATK_ROTATIONS);
}

#[test]
fn rotate_inplace() {
    crate::leveled::tests::test_suite::rotate::test_rotate_inplace(&CTX, ATK_ROTATIONS);
}

#[test]
fn mul_ct_aligned() {
    crate::leveled::tests::test_suite::mul::test_mul_ct_aligned(&CTX);
}
#[test]
fn mul_ct_delta_a_gt_b() {
    crate::leveled::tests::test_suite::mul::test_mul_ct_delta_a_gt_b(&CTX);
}

#[test]
fn mul_ct_delta_a_lt_b() {
    crate::leveled::tests::test_suite::mul::test_mul_ct_delta_a_lt_b(&CTX);
}

#[test]
fn mul_ct_delta_log_decimal() {
    crate::leveled::tests::test_suite::mul::test_mul_ct_delta_log_decimal(&CTX);
}

#[test]
fn mul_ct_smaller_output() {
    crate::leveled::tests::test_suite::mul::test_mul_ct_smaller_output(&CTX);
}

#[test]
fn mul_ct_inplace_aligned() {
    crate::leveled::tests::test_suite::mul::test_mul_ct_inplace_aligned(&CTX);
}

#[test]
fn mul_ct_inplace_self_lt() {
    crate::leveled::tests::test_suite::mul::test_mul_ct_inplace_self_lt(&CTX);
}

#[test]
fn mul_ct_inplace_self_gt() {
    crate::leveled::tests::test_suite::mul::test_mul_ct_inplace_self_gt(&CTX);
}

#[test]
fn square_aligned() {
    crate::leveled::tests::test_suite::mul::test_square_aligned(&CTX);
}

#[test]
fn square_rescaled_input() {
    crate::leveled::tests::test_suite::mul::test_square_rescaled_input(&CTX);
}

#[test]
fn square_inplace() {
    crate::leveled::tests::test_suite::mul::test_square_inplace(&CTX);
}

#[test]
fn square_smaller_output() {
    crate::leveled::tests::test_suite::mul::test_square_smaller_output(&CTX);
}

#[test]
fn mul_pt_znx_aligned() {
    crate::leveled::tests::test_suite::mul::test_mul_pt_znx_aligned(&CTX);
}

#[test]
fn mul_pt_znx_delta_log_decimal() {
    crate::leveled::tests::test_suite::mul::test_mul_pt_znx_delta_log_decimal(&CTX);
}

#[test]
fn mul_pt_znx_smaller_output() {
    crate::leveled::tests::test_suite::mul::test_mul_pt_znx_smaller_output(&CTX);
}

#[test]
fn mul_pt_znx_inplace() {
    crate::leveled::tests::test_suite::mul::test_mul_pt_znx_inplace(&CTX);
}

#[test]
fn mul_pt_rnx_aligned() {
    crate::leveled::tests::test_suite::mul::test_mul_pt_rnx_aligned(&CTX);
}

#[test]
fn mul_pt_rnx_delta_log_decimal() {
    crate::leveled::tests::test_suite::mul::test_mul_pt_rnx_delta_log_decimal(&CTX);
}

#[test]
fn mul_pt_rnx_smaller_output() {
    crate::leveled::tests::test_suite::mul::test_mul_pt_rnx_smaller_output(&CTX);
}

#[test]
fn mul_pt_rnx_inplace() {
    crate::leveled::tests::test_suite::mul::test_mul_pt_rnx_inplace(&CTX);
}

#[test]
fn mul_ct_explicit_metadata_error() {
    crate::leveled::tests::test_suite::mul::test_mul_ct_explicit_metadata_error(&CTX);
}

#[test]
fn mul_pow2_aligned() {
    crate::leveled::tests::test_suite::pow2::test_mul_pow2_aligned(&CTX);
}

#[test]
fn mul_pow2_smaller_output() {
    crate::leveled::tests::test_suite::pow2::test_mul_pow2_smaller_output(&CTX);
}

#[test]
fn mul_pow2_inplace() {
    crate::leveled::tests::test_suite::pow2::test_mul_pow2_inplace(&CTX);
}

#[test]
fn div_pow2_aligned() {
    crate::leveled::tests::test_suite::pow2::test_div_pow2_aligned(&CTX);
}

#[test]
fn div_pow2_smaller_output() {
    crate::leveled::tests::test_suite::pow2::test_div_pow2_smaller_output(&CTX);
}

#[test]
fn div_pow2_inplace() {
    crate::leveled::tests::test_suite::pow2::test_div_pow2_inplace(&CTX);
}

#[test]
fn div_pow2_inplace_explicit_error() {
    crate::leveled::tests::test_suite::pow2::test_div_pow2_inplace_explicit_error(&CTX);
}

#[test]
fn composition_linear_sum() {
    crate::leveled::tests::test_suite::composition::test_linear_sum(&CTX);
}

#[test]
fn composition_poly2_sum() {
    crate::leveled::tests::test_suite::composition::test_poly2_sum(&CTX);
}

#[test]
fn composition_poly2_sum_with_const() {
    crate::leveled::tests::test_suite::composition::test_poly2_sum_with_const(&CTX);
}

#[test]
fn composition_poly2_mul() {
    crate::leveled::tests::test_suite::composition::test_poly2_mul(&CTX);
}

#[test]
fn composition_repeated_square_exhausts_capacity() {
    crate::leveled::tests::test_suite::composition::test_repeated_square_exhausts_capacity(&CTX);
}
