use crate::{
    CKKSCompositionError, CKKSInfos,
    layouts::{ciphertext::CKKSMaintainOps, plaintext::alloc_pt_znx},
    leveled::operations::add::CKKSAddOps,
};
use poulpy_core::layouts::LWEInfos;
use poulpy_hal::api::ScratchOwnedBorrow;

use super::helpers::{TestAddBackend as Backend, TestContext, TestScalar, assert_ckks_error};

pub fn test_reallocate_limbs_checked_error<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let requested_limbs = (ct.max_k().as_usize() - ct.log_decimal()) / ct.base2k().as_usize() + 1;
    let err = ctx
        .module
        .ckks_reallocate_limbs_checked(&mut ct, requested_limbs)
        .unwrap_err();
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

pub fn test_add_pt_znx_alignment_error<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    ct.meta.log_hom_rem = 0;
    let pt_znx = alloc_pt_znx(ctx.degree(), ctx.base2k(), ctx.meta());
    let err = ctx
        .module
        .ckks_add_pt_vec_znx_inplace(&mut ct, &pt_znx, scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "add_pt_znx_alignment",
        &err,
        CKKSCompositionError::PlaintextAlignmentImpossible {
            op: "ckks_add_pt_vec_znx",
            ct_log_hom_rem: 0,
            pt_log_decimal: ctx.meta().log_decimal,
            pt_max_k: pt_znx.max_k().as_usize(),
        },
    );
}
