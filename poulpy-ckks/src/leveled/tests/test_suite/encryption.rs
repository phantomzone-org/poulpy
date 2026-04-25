//! Encrypt / decrypt round-trip tests.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_encrypt_decrypt`] | legacy helper round-trip |
//! | [`test_decrypt_extract_same_meta`] | `available == pt.max_k()`, no truncation |
//! | [`test_decrypt_extract_truncates_log_hom_rem`] | `ct.log_hom_rem() > pt.log_hom_rem()` |
//! | [`test_decrypt_extract_rsh_for_smaller_log_decimal`] | `available < pt.max_k()` → `vec_znx_rsh` |
//! | [`test_decrypt_extract_lsh_for_larger_log_decimal`] | `available > pt.max_k()` → `vec_znx_lsh` |
//! | [`test_decrypt_extract_output_hom_rem_too_large`] | `ct.log_hom_rem() < pt.log_hom_rem()` error |
//! | [`test_decrypt_extract_base2k_mismatch_error`] | plaintext/ciphertext `base2k` mismatch |

use super::helpers::{TestCiphertextBackend as Backend, TestContext, TestScalar, assert_ckks_error, assert_ct_meta};
use crate::{CKKSCompositionError, CKKSInfos, CKKSMeta, layouts::plaintext::alloc_pt_znx, leveled::encryption::CKKSDecrypt};
use poulpy_core::layouts::LWEInfos;
use poulpy_hal::api::ScratchOwnedBorrow;

fn extract_src_prec<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) -> CKKSMeta {
    if ctx.base2k().as_usize() == 19 {
        CKKSMeta {
            log_decimal: 40,
            log_hom_rem: 17,
        }
    } else {
        CKKSMeta {
            log_decimal: 40,
            log_hom_rem: 12,
        }
    }
}

fn extract_fixture<BE: Backend, F: TestScalar>(
    ctx: &TestContext<BE, F>,
    scratch: &mut poulpy_hal::layouts::Scratch<BE>,
) -> crate::layouts::CKKSCiphertext<Vec<u8>> {
    let src_prec = extract_src_prec(ctx);
    ctx.encrypt_with_prec(src_prec.effective_k(), &ctx.re1, &ctx.im1, src_prec, scratch)
}

fn assert_decrypt_extract_success<BE: Backend, F: TestScalar>(label: &str, ctx: &TestContext<BE, F>, dst_prec: CKKSMeta)
where
    poulpy_hal::layouts::Module<BE>: CKKSDecrypt<BE>,
    poulpy_hal::layouts::Scratch<BE>: poulpy_core::ScratchTakeCore<BE>,
{
    let src_prec = extract_src_prec(ctx);
    let mut scratch = ctx.alloc_scratch();
    let ct = extract_fixture(ctx, scratch.borrow());
    assert_ct_meta(&format!("{label} src"), &ct, src_prec.log_decimal, src_prec.log_hom_rem);

    let pt = ctx.decrypt_with_prec(&ct, dst_prec, scratch.borrow()).unwrap();
    assert_eq!(pt.meta, dst_prec, "{label}: decrypt changed destination metadata");

    let (re_out, im_out) = ctx.decode_pt_znx(&pt);
    let (want_prec, assert_log_decimal) = if dst_prec.log_decimal > src_prec.log_decimal {
        // A left-shift during extraction only repacks the same source
        // quantization at a larger scale; it does not manufacture additional
        // absolute precision.
        (src_prec, src_prec.log_decimal)
    } else {
        (dst_prec, dst_prec.log_decimal)
    };
    let (want_re, want_im) = ctx.quantized_slots(&ctx.re1, &ctx.im1, want_prec);
    ctx.assert_precision_for_log_decimal(&format!("{label} re"), &re_out, &want_re, assert_log_decimal);
    ctx.assert_precision_for_log_decimal(&format!("{label} im"), &im_out, &want_im, assert_log_decimal);
}

/// Verifies that encrypt → decrypt → decode recovers the original message.
pub fn test_encrypt_decrypt<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    assert_ct_meta(
        "encrypt_decrypt",
        &ct,
        ctx.meta().log_decimal,
        ctx.max_k() - ctx.meta().log_decimal,
    );
    let (re_out, im_out) = ctx.decrypt_decode(&ct, scratch.borrow());
    ctx.assert_precision_for_log_decimal("encrypt_decrypt re", &re_out, &ctx.re1, ct.log_decimal());
    ctx.assert_precision_for_log_decimal("encrypt_decrypt im", &im_out, &ctx.im1, ct.log_decimal());
}

pub fn test_decrypt_extract_same_meta<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    assert_decrypt_extract_success("decrypt_extract_same_meta", ctx, extract_src_prec(ctx));
}

pub fn test_decrypt_extract_truncates_log_hom_rem<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let src_prec = extract_src_prec(ctx);
    assert_decrypt_extract_success(
        "decrypt_extract_truncates_log_hom_rem",
        ctx,
        CKKSMeta {
            log_decimal: src_prec.log_decimal,
            log_hom_rem: 0,
        },
    );
}

pub fn test_decrypt_extract_rsh_for_smaller_log_decimal<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let src_prec = extract_src_prec(ctx);
    assert_decrypt_extract_success(
        "decrypt_extract_rsh",
        ctx,
        CKKSMeta {
            log_decimal: src_prec.log_decimal - 8,
            log_hom_rem: src_prec.log_hom_rem,
        },
    );
}

pub fn test_decrypt_extract_lsh_for_larger_log_decimal<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let src_prec = extract_src_prec(ctx);
    assert_decrypt_extract_success(
        "decrypt_extract_lsh",
        ctx,
        CKKSMeta {
            log_decimal: src_prec.log_decimal,
            log_hom_rem: src_prec.log_hom_rem - 8,
        },
    );
}

pub fn test_decrypt_extract_output_hom_rem_too_large<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let src_prec = extract_src_prec(ctx);
    let mut scratch = ctx.alloc_scratch();
    let ct = extract_fixture(ctx, scratch.borrow());
    let mut pt = alloc_pt_znx(
        ctx.degree(),
        ctx.base2k(),
        CKKSMeta {
            log_decimal: src_prec.log_decimal,
            log_hom_rem: src_prec.log_hom_rem + 1,
        },
    );
    let err = ctx.module.ckks_decrypt(&mut pt, &ct, &ctx.sk, scratch.borrow()).unwrap_err();
    assert_ckks_error(
        "decrypt_extract_output_hom_rem_too_large",
        &err,
        CKKSCompositionError::PlaintextAlignmentImpossible {
            op: "ckks_extract_pt_znx",
            ct_log_hom_rem: src_prec.log_hom_rem,
            pt_log_decimal: src_prec.log_decimal,
            pt_max_k: pt.max_k().as_usize(),
        },
    );
}

pub fn test_decrypt_extract_base2k_mismatch_error<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let src_prec = extract_src_prec(ctx);
    let mut scratch = ctx.alloc_scratch();
    let ct = extract_fixture(ctx, scratch.borrow());
    let mismatched_base2k = (ctx.base2k().as_usize() / 2).into();
    let mut pt = alloc_pt_znx(ctx.degree(), mismatched_base2k, src_prec);
    let err = ctx.module.ckks_decrypt(&mut pt, &ct, &ctx.sk, scratch.borrow()).unwrap_err();
    assert_ckks_error(
        "decrypt_extract_base2k_mismatch",
        &err,
        CKKSCompositionError::PlaintextBase2KMismatch {
            op: "ckks_extract_pt_znx",
            ct_base2k: ctx.base2k().as_usize(),
            pt_base2k: mismatched_base2k.as_usize(),
        },
    );
}
