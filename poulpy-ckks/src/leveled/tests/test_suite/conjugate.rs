//! Conjugation tests (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer conjugation (`GLWE<_, CKKS>::conjugate`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_conjugate_aligned`] | out-of-place conjugation |
//! | [`test_conjugate_smaller_output`] | out-of-place conjugation |
//!
//! ## Operations-layer conjugation (`GLWE<_, CKKS>::conjugate_inplace`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_conjugate_inplace`] | in-place conjugation |

use crate::{CKKSInfos, leveled::operations::conjugate::CKKSConjugateOps};

use super::helpers::{TestContext, TestRotateBackend as Backend, assert_ct_meta, assert_unary_output_meta};
use poulpy_hal::api::ScratchOwnedBorrow;

// ─── conjugation out-of-place (GLWE<_, CKKS>::conjugate) ───────────────────

/// Conjugation out-of-place: real part preserved, imaginary part negated.
pub fn test_conjugate_aligned<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_conjugate();
    let conj_key = ctx.atk(-1);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_conjugate(&mut ct_res, &ct1, conj_key, scratch.borrow())
        .unwrap();
    assert_unary_output_meta("conjugate", &ct_res, &ct1);
    ctx.assert_decrypt_precision("conjugate", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// Conjugation out-of-place: real part preserved, imaginary part negated.
pub fn test_conjugate_smaller_output<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_conjugate();
    let conj_key = ctx.atk(-1);
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module
        .ckks_conjugate(&mut ct_res, &ct1, conj_key, scratch.borrow())
        .unwrap();
    assert_unary_output_meta("conjugate smaller_output", &ct_res, &ct1);
    ctx.assert_decrypt_precision("conjugate", &ct_res, &want_re, &want_im, scratch.borrow());
}

// ─── conjugation in-place (GLWE<_, CKKS>::conjugate_inplace) ───────────────

/// Conjugation in-place: real part preserved, imaginary part negated.
pub fn test_conjugate_inplace<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_conjugate();
    let conj_key = ctx.atk(-1);
    let expected_log_decimal = ct.log_decimal();
    let expected_log_hom_rem = ct.log_hom_rem();
    ctx.module.ckks_conjugate_inplace(&mut ct, conj_key, scratch.borrow());
    assert_ct_meta("conjugate_inplace", &ct, expected_log_decimal, expected_log_hom_rem);
    ctx.assert_decrypt_precision("conjugate_inplace", &ct, &want_re, &want_im, scratch.borrow());
}
