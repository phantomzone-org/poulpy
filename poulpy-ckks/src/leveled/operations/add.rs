//! CKKS ciphertext addition.
//!
//! Provides ct+ct, ct+pt (compact and prepared), and ct+constant variants,
//! each in out-of-place and in-place form.  Compact plaintext operands are
//! expanded into the ciphertext torus layout via [`fill_offset_pt`] before
//! the underlying GLWE addition.
//!
//! [`fill_offset_pt`]: super::utils::fill_offset_pt

use super::{
    align::{align_to, are_cts_aligned, assert_cts_aligned, common_window},
    utils::{const_pt_from_scratch, const_pt_scratch_bytes, offset_pt_from_scratch, offset_pt_scratch_bytes},
};
use crate::layouts::{
    ciphertext::{CKKSCiphertext, CKKSCiphertextToRef},
    plaintext::CKKSPlaintext,
    plaintext_prepared::CKKSPlaintextPrepared,
};
use poulpy_core::{
    GLWEAdd, GLWEAlign, ScratchTakeCore,
    layouts::{GLWE, GLWEInfos, GLWELayout, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

/// Returns the scratch bytes needed for [`add`] and [`add_inplace`].
pub fn add_tmp_bytes<BE: Backend>(
    module: &Module<BE>,
    a: &CKKSCiphertext<impl DataRef>,
    b: &CKKSCiphertext<impl DataRef>,
) -> usize
where
    Module<BE>: GLWEAlign<BE>,
{
    let (_, target_k) = common_window(a, b);
    let layout = GLWELayout {
        n: a.inner.n(),
        base2k: a.inner.base2k(),
        k: target_k,
        rank: a.inner.rank(),
    };
    GLWE::bytes_of_from_infos(&layout)
        + module
            .glwe_align_tmp_bytes(&layout, &a.inner)
            .max(module.glwe_align_tmp_bytes(&layout, &b.inner))
}

/// Computes `res = a + b` for already-aligned ciphertexts (no scratch needed).
pub fn add_aligned<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
    b: &CKKSCiphertext<impl DataRef>,
) where
    Module<BE>: GLWEAdd,
{
    a.assert_valid("add_aligned lhs");
    b.assert_valid("add_aligned rhs");
    assert_cts_aligned(a, b, "add_aligned");
    assert_eq!(
        a.torus_scale_bits(),
        b.torus_scale_bits(),
        "add_aligned: torus_scale_bits mismatch ({} != {})",
        a.torus_scale_bits(),
        b.torus_scale_bits()
    );
    res.torus_scale_bits = a.torus_scale_bits();
    res.offset_bits = a.offset_bits();
    res.set_active_k(a.inner.k());
    res.zero_inactive_tail();
    module.glwe_add(&mut res.inner, &a.inner, &b.inner);
    res.assert_valid("add_aligned result");
}

/// Computes `res = a + b`.
pub fn add<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
    b: &CKKSCiphertext<impl DataRef>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEAdd + GLWEAlign<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    a.assert_valid("add lhs");
    b.assert_valid("add rhs");
    assert_eq!(
        a.torus_scale_bits(),
        b.torus_scale_bits(),
        "add: torus_scale_bits mismatch ({} != {})",
        a.torus_scale_bits(),
        b.torus_scale_bits()
    );
    let aligned_inputs = are_cts_aligned(a, b);
    if aligned_inputs {
        add_aligned(module, res, a, b);
        return;
    }

    let (offset_common, target_k) = common_window(a, b);
    let a_needs_align = a.offset_bits() != offset_common || a.prefix_bits() != target_k.0;
    let b_needs_align = b.offset_bits() != offset_common || b.prefix_bits() != target_k.0;

    res.torus_scale_bits = a.torus_scale_bits();
    res.offset_bits = offset_common;
    res.set_active_k(target_k);
    res.zero_inactive_tail();

    if !a_needs_align && !b_needs_align {
        module.glwe_add(&mut res.inner, &a.inner, &b.inner);
        res.assert_valid("add result");
        return;
    }

    let layout = GLWELayout {
        n: a.inner.n(),
        base2k: a.inner.base2k(),
        k: target_k,
        rank: a.inner.rank(),
    };
    let (tmp, scratch_1) = scratch.take_glwe(&layout);
    let mut tmp_ct = CKKSCiphertext {
        inner: tmp,
        offset_bits: offset_common,
        torus_scale_bits: a.torus_scale_bits(),
    };
    align_to(module, &mut tmp_ct, a, offset_common, target_k, scratch_1);
    align_to(module, res, b, offset_common, target_k, scratch_1);
    module.glwe_add_inplace(&mut res.inner, &tmp_ct.inner);
    res.assert_valid("add result");
}

/// Computes `res += a` for already-aligned ciphertexts (no scratch needed).
pub fn add_aligned_inplace<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
) where
    Module<BE>: GLWEAdd,
{
    res.assert_valid("add_aligned_inplace lhs");
    a.assert_valid("add_aligned_inplace rhs");
    assert_cts_aligned(res, a, "add_aligned_inplace");
    assert_eq!(
        res.torus_scale_bits(),
        a.torus_scale_bits(),
        "add_aligned_inplace: torus_scale_bits mismatch ({} != {})",
        res.torus_scale_bits(),
        a.torus_scale_bits()
    );
    module.glwe_add_inplace(&mut res.inner, &a.inner);
    res.assert_valid("add_aligned_inplace result");
}

/// Computes `res += a` in place.
pub fn add_inplace<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    a: &CKKSCiphertext<impl DataRef>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEAdd + GLWEAlign<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    res.assert_valid("add_inplace lhs");
    a.assert_valid("add_inplace rhs");
    assert_eq!(
        res.torus_scale_bits(),
        a.torus_scale_bits(),
        "add_inplace: torus_scale_bits mismatch ({} != {})",
        res.torus_scale_bits(),
        a.torus_scale_bits()
    );
    let aligned_inputs = are_cts_aligned(res, a);
    if aligned_inputs {
        add_aligned_inplace(module, res, a);
        return;
    }

    let (offset_common, target_k) = common_window(res, a);
    let res_needs_align = res.offset_bits() != offset_common || res.prefix_bits() != target_k.0;
    let a_needs_align = a.offset_bits() != offset_common || a.prefix_bits() != target_k.0;

    if !res_needs_align && !a_needs_align {
        module.glwe_add_inplace(&mut res.inner, &a.inner);
        res.offset_bits = offset_common;
        res.assert_valid("add_inplace result");
        return;
    }

    let layout = GLWELayout {
        n: res.inner.n(),
        base2k: res.inner.base2k(),
        k: target_k,
        rank: res.inner.rank(),
    };
    let (tmp, scratch_1) = scratch.take_glwe(&layout);
    let mut tmp_ct = CKKSCiphertext {
        inner: tmp,
        offset_bits: offset_common,
        torus_scale_bits: res.torus_scale_bits(),
    };
    align_to(module, &mut tmp_ct, &res.to_ref(), offset_common, target_k, scratch_1);
    align_to(module, res, a, offset_common, target_k, scratch_1);
    module.glwe_add_inplace(&mut res.inner, &tmp_ct.inner);
    res.assert_valid("add_inplace result");
}

/// Returns the scratch bytes needed for [`add_pt`] and [`add_pt_inplace`].
pub fn add_pt_tmp_bytes<BE: Backend>(module: &Module<BE>, ct: &CKKSCiphertext<impl Data>) -> usize
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

/// Computes `res = ct + pt`.
pub fn add_pt<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    pt: &CKKSPlaintext<impl DataRef>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEAdd + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    ct.assert_valid("add_pt ciphertext");
    assert_eq!(
        ct.torus_scale_bits(),
        pt.embed_bits(),
        "add_pt: scale mismatch (ct.torus_scale_bits={}, pt.embed_bits={})",
        ct.torus_scale_bits(),
        pt.embed_bits()
    );
    res.torus_scale_bits = ct.torus_scale_bits();
    res.offset_bits = ct.offset_bits();
    let (full_pt, _) = offset_pt_from_scratch(
        module,
        ct.inner.n(),
        ct.inner.base2k(),
        poulpy_core::layouts::TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32),
        poulpy_core::layouts::TorusPrecision(ct.offset_bits()),
        pt,
        scratch,
    );
    module.glwe_add(&mut res.inner, &ct.inner, &full_pt);
    res.assert_valid("add_pt result");
}

/// Computes `ct += pt` in place.
pub fn add_pt_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    pt: &CKKSPlaintext<impl DataRef>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEAdd + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    ct.assert_valid("add_pt_inplace ciphertext");
    assert_eq!(
        ct.torus_scale_bits(),
        pt.embed_bits(),
        "add_pt_inplace: scale mismatch (ct.torus_scale_bits={}, pt.embed_bits={})",
        ct.torus_scale_bits(),
        pt.embed_bits()
    );
    let (full_pt, _) = offset_pt_from_scratch(
        module,
        ct.inner.n(),
        ct.inner.base2k(),
        poulpy_core::layouts::TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32),
        poulpy_core::layouts::TorusPrecision(ct.offset_bits()),
        pt,
        scratch,
    );
    module.glwe_add_inplace(&mut ct.inner, &full_pt);
    ct.assert_valid("add_pt_inplace result");
}

/// Computes `res = ct + pt` using a pre-expanded plaintext (no scratch needed).
pub fn add_prepared_pt<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    pt: &CKKSPlaintextPrepared<impl DataRef>,
) where
    Module<BE>: GLWEAdd,
{
    ct.assert_valid("add_prepared_pt ciphertext");
    assert_eq!(
        ct.torus_scale_bits(),
        pt.embed_bits(),
        "add_prepared_pt: scale mismatch (ct.torus_scale_bits={}, pt.embed_bits={})",
        ct.torus_scale_bits(),
        pt.embed_bits()
    );
    res.torus_scale_bits = ct.torus_scale_bits();
    res.offset_bits = ct.offset_bits();
    module.glwe_add(&mut res.inner, &ct.inner, &pt.inner);
    res.assert_valid("add_prepared_pt result");
}

/// Computes `ct += pt` in place using a pre-expanded plaintext (no scratch needed).
pub fn add_prepared_pt_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    pt: &CKKSPlaintextPrepared<impl DataRef>,
) where
    Module<BE>: GLWEAdd,
{
    ct.assert_valid("add_prepared_pt_inplace ciphertext");
    assert_eq!(
        ct.torus_scale_bits(),
        pt.embed_bits(),
        "add_prepared_pt_inplace: scale mismatch (ct.torus_scale_bits={}, pt.embed_bits={})",
        ct.torus_scale_bits(),
        pt.embed_bits()
    );
    module.glwe_add_inplace(&mut ct.inner, &pt.inner);
    ct.assert_valid("add_prepared_pt_inplace result");
}

/// Returns the scratch bytes needed for [`add_const`] and [`add_const_inplace`].
pub fn add_const_tmp_bytes<BE: Backend>(module: &Module<BE>, ct: &CKKSCiphertext<impl Data>) -> usize
where
    Module<BE>: VecZnxNormalizeTmpBytes,
{
    const_pt_scratch_bytes(ct) + add_pt_tmp_bytes(module, ct)
}

/// Computes `res = ct + c` where `c = re + i*im`.
pub fn add_const<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    re: f64,
    im: f64,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEAdd + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let (pt, scratch_rest) = const_pt_from_scratch(ct, re, im, scratch);
    add_pt(module, res, ct, &pt, scratch_rest);
}

/// Computes `ct += c` in place where `c = re + i*im`.
pub fn add_const_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    re: f64,
    im: f64,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEAdd + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let (pt, scratch_rest) = const_pt_from_scratch(ct, re, im, scratch);
    add_pt_inplace(module, ct, &pt, scratch_rest);
}
