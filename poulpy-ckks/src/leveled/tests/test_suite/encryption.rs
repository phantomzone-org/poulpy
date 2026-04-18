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

use super::helpers::{TestCiphertextBackend as Backend, TestContext, assert_ckks_error, assert_ct_meta, assert_precision};
use crate::{CKKSCompositionError, CKKSInfos, CKKSMeta, layouts::plaintext::alloc_pt_znx, leveled::encryption::CKKSDecrypt};
use poulpy_core::layouts::LWEInfos;
use poulpy_hal::api::ScratchOwnedBorrow;

const EXTRACT_SRC_PREC: CKKSMeta = CKKSMeta {
    log_decimal: 40,
    log_hom_rem: 12,
};
const EXTRACT_MIN_BITS: f64 = 18.0;

fn extract_fixture<BE: Backend>(
    ctx: &TestContext<BE>,
    scratch: &mut poulpy_hal::layouts::Scratch<BE>,
) -> crate::layouts::CKKSCiphertext<Vec<u8>> {
    ctx.encrypt_with_prec(EXTRACT_SRC_PREC.effective_k(), &ctx.re1, &ctx.im1, EXTRACT_SRC_PREC, scratch)
}

fn assert_decrypt_extract_success<BE: Backend>(label: &str, ctx: &TestContext<BE>, dst_prec: CKKSMeta)
where
    poulpy_hal::layouts::Module<BE>: CKKSDecrypt<BE>,
    poulpy_hal::layouts::Scratch<BE>: poulpy_core::ScratchTakeCore<BE>,
{
    let mut scratch = ctx.alloc_scratch();
    let ct = extract_fixture(ctx, scratch.borrow());
    assert_ct_meta(
        &format!("{label} src"),
        &ct,
        EXTRACT_SRC_PREC.log_decimal,
        EXTRACT_SRC_PREC.log_hom_rem,
    );

    let pt = ctx.decrypt_with_prec(&ct, dst_prec, scratch.borrow()).unwrap();
    assert_eq!(pt.meta, dst_prec, "{label}: decrypt changed destination metadata");

    let (re_out, im_out) = ctx.decode_pt_znx(&pt);
    let (want_re, want_im) = ctx.quantized_slots(&ctx.re1, &ctx.im1, dst_prec);
    assert_precision(&format!("{label} re"), &re_out, &want_re, EXTRACT_MIN_BITS);
    assert_precision(&format!("{label} im"), &im_out, &want_im, EXTRACT_MIN_BITS);
}

/// Verifies that encrypt → decrypt → decode recovers the original message.
pub fn test_encrypt_decrypt<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    assert_ct_meta(
        "encrypt_decrypt",
        &ct,
        ctx.meta().log_decimal,
        ctx.max_k() - ctx.meta().log_decimal,
    );
    let (re_out, im_out) = ctx.decrypt_decode(&ct, scratch.borrow());
    assert_precision("encrypt_decrypt re", &re_out, &ctx.re1, 20.0);
    assert_precision("encrypt_decrypt im", &im_out, &ctx.im1, 20.0);
}

pub fn test_decrypt_extract_same_meta<BE: Backend>(ctx: &TestContext<BE>) {
    assert_decrypt_extract_success("decrypt_extract_same_meta", ctx, EXTRACT_SRC_PREC);
}

pub fn test_decrypt_extract_truncates_log_hom_rem<BE: Backend>(ctx: &TestContext<BE>) {
    assert_decrypt_extract_success(
        "decrypt_extract_truncates_log_hom_rem",
        ctx,
        CKKSMeta {
            log_decimal: EXTRACT_SRC_PREC.log_decimal,
            log_hom_rem: 0,
        },
    );
}

pub fn test_decrypt_extract_rsh_for_smaller_log_decimal<BE: Backend>(ctx: &TestContext<BE>) {
    assert_decrypt_extract_success(
        "decrypt_extract_rsh",
        ctx,
        CKKSMeta {
            log_decimal: EXTRACT_SRC_PREC.log_decimal - 8,
            log_hom_rem: EXTRACT_SRC_PREC.log_hom_rem,
        },
    );
}

pub fn test_decrypt_extract_lsh_for_larger_log_decimal<BE: Backend>(ctx: &TestContext<BE>) {
    assert_decrypt_extract_success(
        "decrypt_extract_lsh",
        ctx,
        CKKSMeta {
            log_decimal: EXTRACT_SRC_PREC.log_decimal + 8,
            log_hom_rem: EXTRACT_SRC_PREC.log_hom_rem - 8,
        },
    );
}

pub fn test_decrypt_extract_output_hom_rem_too_large<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = extract_fixture(ctx, scratch.borrow());
    let mut pt = alloc_pt_znx(
        ctx.degree(),
        ctx.base2k(),
        CKKSMeta {
            log_decimal: EXTRACT_SRC_PREC.log_decimal,
            log_hom_rem: EXTRACT_SRC_PREC.log_hom_rem + 1,
        },
    );
    let err = ctx.module.ckks_decrypt(&mut pt, &ct, &ctx.sk, scratch.borrow()).unwrap_err();
    assert_ckks_error(
        "decrypt_extract_output_hom_rem_too_large",
        &err,
        CKKSCompositionError::PlaintextAlignmentImpossible {
            op: "ckks_extract_pt_znx",
            ct_log_hom_rem: EXTRACT_SRC_PREC.log_hom_rem,
            pt_log_decimal: EXTRACT_SRC_PREC.log_decimal,
            pt_max_k: pt.max_k().as_usize(),
        },
    );
}

pub fn test_decrypt_extract_base2k_mismatch_error<BE: Backend>(ctx: &TestContext<BE>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = extract_fixture(ctx, scratch.borrow());
    let mismatched_base2k = (ctx.base2k().as_usize() / 2).into();
    let mut pt = alloc_pt_znx(ctx.degree(), mismatched_base2k, EXTRACT_SRC_PREC);
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
