//! CKKS ciphertext subtraction.
//!
//! Provides ct-ct, ct-pt (compact and prepared), and ct-constant variants,
//! each in out-of-place and in-place form.  Compact plaintext operands are
//! expanded into the ciphertext torus layout via [`fill_offset_pt`] before
//! the underlying GLWE subtraction.
//!
//! [`fill_offset_pt`]: super::utils::fill_offset_pt

use super::utils::{const_pt_from_scratch, const_pt_scratch_bytes, offset_pt_from_scratch, offset_pt_scratch_bytes};
use crate::layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext, plaintext_prepared::CKKSPlaintextPrepared};
use poulpy_core::{GLWESub, ScratchTakeCore, layouts::LWEInfos};
use poulpy_hal::{
    api::{VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

/// Computes `res = a - b`.
pub fn sub<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
    b: &CKKSCiphertext<impl DataRef>,
) where
    Module<BE>: GLWESub,
{
    assert_eq!(
        a.log_delta, b.log_delta,
        "sub: log_delta mismatch ({} != {})",
        a.log_delta, b.log_delta
    );
    res.log_delta = a.log_delta;
    module.glwe_sub(&mut res.inner, &a.inner, &b.inner);
}

/// Computes `res -= a` in place.
pub fn sub_inplace<BE: Backend>(module: &Module<BE>, res: &mut CKKSCiphertext<impl DataMut>, a: &CKKSCiphertext<impl DataRef>)
where
    Module<BE>: GLWESub,
{
    assert_eq!(
        res.log_delta, a.log_delta,
        "sub_inplace: log_delta mismatch ({} != {})",
        res.log_delta, a.log_delta
    );
    module.glwe_sub_inplace(&mut res.inner, &a.inner);
}

/// Returns the scratch bytes needed for [`sub_pt`] and [`sub_pt_inplace`].
pub fn sub_pt_tmp_bytes<BE: Backend>(module: &Module<BE>, ct: &CKKSCiphertext<impl Data>) -> usize
where
    Module<BE>: VecZnxNormalizeTmpBytes,
{
    offset_pt_scratch_bytes(
        module,
        ct.inner.n(),
        ct.inner.base2k(),
        poulpy_core::layouts::TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32),
    )
}

/// Computes `res = ct - pt`.
pub fn sub_pt<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    pt: &CKKSPlaintext<impl DataRef>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWESub + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    assert_eq!(
        ct.log_delta, pt.log_delta,
        "sub_pt: log_delta mismatch (ct={}, pt={})",
        ct.log_delta, pt.log_delta
    );
    res.log_delta = ct.log_delta;
    let (full_pt, _) = offset_pt_from_scratch(
        module,
        ct.inner.n(),
        ct.inner.base2k(),
        poulpy_core::layouts::TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32),
        ct.inner.k(),
        pt,
        scratch,
    );
    module.glwe_sub(&mut res.inner, &ct.inner, &full_pt);
}

/// Computes `ct -= pt` in place.
pub fn sub_pt_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    pt: &CKKSPlaintext<impl DataRef>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWESub + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    assert_eq!(
        ct.log_delta, pt.log_delta,
        "sub_pt_inplace: log_delta mismatch (ct={}, pt={})",
        ct.log_delta, pt.log_delta
    );
    let (full_pt, _) = offset_pt_from_scratch(
        module,
        ct.inner.n(),
        ct.inner.base2k(),
        poulpy_core::layouts::TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32),
        ct.inner.k(),
        pt,
        scratch,
    );
    module.glwe_sub_inplace(&mut ct.inner, &full_pt);
}

/// Computes `res = ct - pt` using a pre-expanded plaintext (no scratch needed).
pub fn sub_prepared_pt<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    pt: &CKKSPlaintextPrepared<impl DataRef>,
) where
    Module<BE>: GLWESub,
{
    assert_eq!(
        ct.log_delta, pt.log_delta,
        "sub_prepared_pt: log_delta mismatch (ct={}, pt={})",
        ct.log_delta, pt.log_delta
    );
    res.log_delta = ct.log_delta;
    module.glwe_sub(&mut res.inner, &ct.inner, &pt.inner);
}

/// Computes `ct -= pt` in place using a pre-expanded plaintext (no scratch needed).
pub fn sub_prepared_pt_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    pt: &CKKSPlaintextPrepared<impl DataRef>,
) where
    Module<BE>: GLWESub,
{
    assert_eq!(
        ct.log_delta, pt.log_delta,
        "sub_prepared_pt_inplace: log_delta mismatch (ct={}, pt={})",
        ct.log_delta, pt.log_delta
    );
    module.glwe_sub_inplace(&mut ct.inner, &pt.inner);
}

/// Returns the scratch bytes needed for [`sub_const`] and [`sub_const_inplace`].
pub fn sub_const_tmp_bytes<BE: Backend>(module: &Module<BE>, ct: &CKKSCiphertext<impl Data>) -> usize
where
    Module<BE>: VecZnxNormalizeTmpBytes,
{
    const_pt_scratch_bytes(ct) + sub_pt_tmp_bytes(module, ct)
}

/// Computes `res = ct - c` where `c = re + i*im`.
pub fn sub_const<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    re: f64,
    im: f64,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWESub + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let (pt, scratch_rest) = const_pt_from_scratch(ct, re, im, scratch);
    sub_pt(module, res, ct, &pt, scratch_rest);
}

/// Computes `ct -= c` in place where `c = re + i*im`.
pub fn sub_const_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    re: f64,
    im: f64,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWESub + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let (pt, scratch_rest) = const_pt_from_scratch(ct, re, im, scratch);
    sub_pt_inplace(module, ct, &pt, scratch_rest);
}
