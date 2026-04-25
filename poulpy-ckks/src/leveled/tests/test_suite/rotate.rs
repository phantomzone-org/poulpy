//! Slot rotation tests (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer rotation (`GLWE<_, CKKS>::rotate`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_rotate_aligned`] | out-of-place rotation for each requested shift |
//!
//! ## Operations-layer rotation (`GLWE<_, CKKS>::rotate_inplace`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_rotate_inplace`] | in-place rotation for each requested shift |

use crate::{CKKSInfos, leveled::operations::rotate::CKKSRotateOps};
use std::collections::HashMap;

use super::helpers::{
    TestContext, TestRotateBackend as Backend, TestScalar, assert_ckks_error, assert_ct_meta, assert_unary_output_meta,
};
use poulpy_core::layouts::GLWEAutomorphismKeyPrepared;
use poulpy_hal::api::ScratchOwnedBorrow;
use poulpy_hal::layouts::DeviceBuf;

// ─── rotation out-of-place (GLWE<_, CKKS>::rotate) ─────────────────────────

/// Rotation out-of-place: slot values are cyclically shifted.
pub fn test_rotate_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>, rotations: &[i64]) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    for &r in rotations {
        let (want_re, want_im) = ctx.want_rotate(r);
        let mut ct_res = ctx.alloc_ct(ctx.max_k());
        ctx.module
            .ckks_rotate(&mut ct_res, &ct, r, ctx.atks(), scratch.borrow())
            .unwrap();
        assert_unary_output_meta(&format!("rotate({r})"), &ct_res, &ct);
        ctx.assert_decrypt_precision(&format!("rotate({r})"), &ct_res, &want_re, &want_im, scratch.borrow());
    }
}

/// Rotation out-of-place: slot values are cyclically shifted.
pub fn test_rotate_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>, rotations: &[i64]) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    for &r in rotations {
        let (want_re, want_im) = ctx.want_rotate(r);
        let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
        ctx.module
            .ckks_rotate(&mut ct_res, &ct, r, ctx.atks(), scratch.borrow())
            .unwrap();
        assert_unary_output_meta(&format!("rotate smaller_output({r})"), &ct_res, &ct);
        ctx.assert_decrypt_precision(&format!("rotate({r})"), &ct_res, &want_re, &want_im, scratch.borrow());
    }
}

// ─── rotation in-place (GLWE<_, CKKS>::rotate_inplace) ─────────────────────

/// Rotation in-place: slot values are cyclically shifted.
pub fn test_rotate_inplace<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>, rotations: &[i64]) {
    let mut scratch = ctx.alloc_scratch();
    for &r in rotations {
        let (want_re, want_im) = ctx.want_rotate(r);
        let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
        let expected_log_decimal = ct.log_decimal();
        let expected_log_hom_rem = ct.log_hom_rem();
        ctx.module
            .ckks_rotate_inplace(&mut ct, r, ctx.atks(), scratch.borrow())
            .unwrap();
        assert_ct_meta(
            &format!("rotate_inplace({r})"),
            &ct,
            expected_log_decimal,
            expected_log_hom_rem,
        );
        ctx.assert_decrypt_precision(&format!("rotate_inplace({r})"), &ct, &want_re, &want_im, scratch.borrow());
    }
}

pub fn test_rotate_inplace_missing_key_error<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let empty_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<DeviceBuf<BE>, BE>> = HashMap::new();
    let err = ctx
        .module
        .ckks_rotate_inplace(&mut ct, 1, &empty_keys, scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "rotate_inplace missing_key",
        &err,
        crate::CKKSCompositionError::MissingAutomorphismKey {
            op: "rotate_inplace",
            rotation: 1,
        },
    );
}
