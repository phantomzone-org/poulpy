//! Addition tests for the `CKKSAddOpsUnsafe` unsafe API.
//!
//! The safe [`CKKSAddOps`](super::super::add::CKKSAddOps) path is literally
//! the unnormalized default plus a trailing `glwe_normalize_assign`, so
//! the many path-coverage tests in [`super::add`] already exercise the
//! shared core for free. These tests only cover what's structurally unique
//! to the unsafe API:
//!
//! - the `unsafe`-trait dispatch reaches the right default helper,
//! - meta (`log_budget`, `log_delta`) is set by the unnormalized op,
//! - a caller-supplied `glwe_normalize_assign` recovers a decryptable
//!   ciphertext equivalent to the safe path.
//!
//! One test is kept per distinct kernel family:
//!
//! | Function | Kernel exercised |
//! |----------|------------------|
//! | [`test_add_ct_aligned_unsafe`] | ct+ct, `glwe_add_into` / shift-add fast path |
//! | [`test_add_ct_assign_aligned_unsafe`] | ct+ct inplace, `glwe_add_assign` |
//! | [`test_add_pt_vec_znx_into_aligned_unsafe`] | ct + ZNX plaintext, `VecZnxRshAddIntoBackend` |
//! | [`test_add_const_znx_into_aligned_unsafe`] | ct + ZNX const, raw `data_mut()[..] += digit` path |

use poulpy_core::{GLWENormalize, layouts::GLWEToBackendMut};
use poulpy_hal::api::ScratchOwnedBorrow;

use crate::{CKKSInfos, leveled::api::CKKSAddOpsUnsafe};

use super::helpers::{
    TestAddBackend as Backend, TestContext, TestScalar, assert_binary_output_meta, assert_ct_meta, assert_unary_output_meta,
};

const CONST_RE: f64 = 0.314_159_265_358_979_3;
const CONST_IM: f64 = -0.271_828_182_845_904_5;

pub fn test_add_ct_aligned_unsafe<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, &mut scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, &mut scratch.borrow());
    let (want_re, want_im) = ctx.want_add();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    unsafe {
        ctx.module
            .ckks_add_into_unsafe(&mut ct_res, &ct1, &ct2, &mut scratch.borrow())
            .unwrap();
    }
    assert_binary_output_meta("add_ct_aligned_unsafe", &ct_res, &ct1, &ct2);
    ctx.module.glwe_normalize_assign(
        &mut <crate::layouts::CKKSCiphertext<Vec<u8>> as GLWEToBackendMut<BE>>::to_backend_mut(&mut ct_res),
        &mut scratch.borrow(),
    );
    ctx.assert_decrypt_precision("add_ct_aligned_unsafe", &ct_res, &want_re, &want_im, &mut scratch.borrow());
}

pub fn test_add_ct_assign_aligned_unsafe<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, &mut scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, &mut scratch.borrow());
    let (want_re, want_im) = ctx.want_add();
    let expected_log_budget = ct1.log_budget().min(ct2.log_budget());
    let expected_log_delta = ct1.log_delta().max(ct2.log_delta());
    unsafe {
        ctx.module
            .ckks_add_assign_unsafe(&mut ct1, &ct2, &mut scratch.borrow())
            .unwrap();
    }
    assert_ct_meta("add_ct_assign_aligned_unsafe", &ct1, expected_log_delta, expected_log_budget);
    ctx.module.glwe_normalize_assign(
        &mut <crate::layouts::CKKSCiphertext<Vec<u8>> as GLWEToBackendMut<BE>>::to_backend_mut(&mut ct1),
        &mut scratch.borrow(),
    );
    ctx.assert_decrypt_precision(
        "add_ct_assign_aligned_unsafe",
        &ct1,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

pub fn test_add_pt_vec_znx_into_aligned_unsafe<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, &mut scratch.borrow());
    let pt_znx = ctx.encode_pt_znx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_add();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    unsafe {
        ctx.module
            .ckks_add_pt_vec_znx_into_unsafe(&mut ct_res, &ct1, &pt_znx, &mut scratch.borrow())
            .unwrap();
    }
    assert_unary_output_meta("add_pt_vec_znx_unsafe", &ct_res, &ct1);
    ctx.module.glwe_normalize_assign(
        &mut <crate::layouts::CKKSCiphertext<Vec<u8>> as GLWEToBackendMut<BE>>::to_backend_mut(&mut ct_res),
        &mut scratch.borrow(),
    );
    ctx.assert_decrypt_precision("add_pt_vec_znx_unsafe", &ct_res, &want_re, &want_im, &mut scratch.borrow());
}

pub fn test_add_const_znx_into_aligned_unsafe<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, &mut scratch.borrow());
    let (const_re, const_im) = ctx.quantized_const(CONST_RE, CONST_IM, ctx.meta().log_delta);
    let (want_re, want_im) = ctx.want_add_const_from(&ctx.re1, &ctx.im1, const_re, const_im);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    let cst_znx = ctx.const_rnx(Some(CONST_RE), Some(CONST_IM), ctx.meta());
    unsafe {
        ctx.module
            .ckks_add_pt_const_znx_into_unsafe(&mut ct_res, &ct, 0, &cst_znx, 0, &mut scratch.borrow())
            .unwrap();
    }
    assert_unary_output_meta("add_const_znx_into_aligned_unsafe", &ct_res, &ct);
    ctx.module.glwe_normalize_assign(
        &mut <crate::layouts::CKKSCiphertext<Vec<u8>> as GLWEToBackendMut<BE>>::to_backend_mut(&mut ct_res),
        &mut scratch.borrow(),
    );
    ctx.assert_decrypt_precision(
        "add_const_znx_into_aligned_unsafe",
        &ct_res,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}
