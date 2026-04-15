use crate::{
    CKKSCompositionError, CKKSInfos,
    layouts::{ciphertext::CKKSMaintainOps, plaintext::alloc_pt_znx},
    leveled::operations::add::CKKSAddOps,
};
use poulpy_core::layouts::LWEInfos;
use poulpy_hal::api::ScratchOwnedBorrow;

use super::helpers::{TestAddBackend as Backend, TestContext, assert_ckks_error};

pub fn test_set_log_decimal_out_of_range_error<BE: Backend>(ctx: &TestContext<BE>) {
    let mut ct = ctx.alloc_ct(ctx.max_k());
    let max_k = ct.max_k().as_usize();
    let requested_log_decimal = max_k + 1;
    let err = ct.set_log_decimal(requested_log_decimal).unwrap_err();
    assert_ckks_error(
        "set_log_decimal_out_of_range",
        &err,
        CKKSCompositionError::LogDecimalOutOfRange {
            max_k,
            log_hom_rem: 0,
            requested_log_decimal,
        },
    );
}

pub fn test_set_log_hom_rem_out_of_range_error<BE: Backend>(ctx: &TestContext<BE>) {
    let mut ct = ctx.alloc_ct(ctx.max_k());
    let max_k = ct.max_k().as_usize();
    let requested_log_hom_rem = max_k + 1;
    let err = ct.set_log_hom_rem(requested_log_hom_rem).unwrap_err();
    assert_ckks_error(
        "set_log_hom_rem_out_of_range",
        &err,
        CKKSCompositionError::LogHomRemOutOfRange {
            max_k,
            log_decimal: 0,
            requested_log_hom_rem,
        },
    );
}

pub fn test_reallocate_limbs_checked_error<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let requested_limbs = (ct.max_k().as_usize() - ct.log_decimal()) / ct.base2k().as_usize() + 1;
    let err = ct.reallocate_limbs_checked(requested_limbs).unwrap_err();
    assert_ckks_error(
        "reallocate_limbs_checked",
        &err,
        CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
            max_k: ct.max_k().as_usize(),
            log_decimal: ct.log_decimal(),
            base2k: ct.base2k().as_usize(),
            requested_limbs,
        },
    );
}

pub fn test_add_pt_znx_alignment_error<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    ct.set_log_hom_rem(0).unwrap();
    let pt_znx = alloc_pt_znx(ctx.degree(), ctx.base2k(), ctx.meta());
    let err = ct.add_pt_znx_inplace(&ctx.module, &pt_znx, scratch.borrow()).unwrap_err();
    assert_ckks_error(
        "add_pt_znx_alignment",
        &err,
        CKKSCompositionError::PlaintextAlignmentImpossible {
            op: "ckks_add_pt_znx",
            ct_log_hom_rem: 0,
            pt_log_decimal: ctx.meta().log_decimal,
            pt_max_k: pt_znx.max_k().as_usize(),
        },
    );
}
