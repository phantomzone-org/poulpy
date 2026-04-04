//! Shared helpers for CKKS leveled operations.
//!
//! This module keeps only the placement/sign-extension logic that is reused
//! across encryption and ciphertext/plaintext arithmetic.

use crate::layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext};
use poulpy_core::{
    ScratchTakeCore,
    layouts::{Base2K, Degree, GLWEPlaintext, GLWEPlaintextLayout, LWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::{VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, ZnxInfos, ZnxView, ZnxViewMut},
};

/// Expands a compact CKKS plaintext into a full torus plaintext by placing the
/// compact integer inside the target `k`-bit torus window.
pub(crate) fn fill_offset_pt<BE: Backend>(
    module: &Module<BE>,
    full: &mut GLWEPlaintext<impl DataMut>,
    target_k: TorusPrecision,
    pt: &CKKSPlaintext<impl DataRef>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: VecZnxNormalize<BE>,
{
    assert_eq!(full.base2k.0, pt.inner.base2k.0, "fill_offset_pt: base2k mismatch");
    assert!(
        pt.inner.data.size() <= full.data.size(),
        "plaintext has more limbs than ciphertext"
    );
    let offset_bits = (pt.inner.data.size() * pt.inner.base2k.0 as usize) as i64 - target_k.0 as i64;
    if offset_bits % full.base2k.0 as i64 == 0 {
        let offset_limbs = (-offset_bits / full.base2k.0 as i64) as usize;
        assert!(
            offset_limbs + pt.inner.data.size() <= full.data.size(),
            "fill_offset_pt: offset out of range"
        );
        full.data.raw_mut().fill(0);
        for l in 0..pt.inner.data.size() {
            let src_limb: &[i64] = pt.inner.data.at(0, l);
            let dst_limb: &mut [i64] = full.data.at_mut(0, offset_limbs + l);
            dst_limb.copy_from_slice(src_limb);
        }
        return;
    }

    module.vec_znx_normalize(
        &mut full.data,
        full.base2k.0 as usize,
        offset_bits,
        0,
        &pt.inner.data,
        pt.inner.base2k.0 as usize,
        0,
        scratch,
    );
}

/// Compresses a full torus plaintext back into the compact CKKS representation
/// using the inverse placement shift for the same target message position.
pub(crate) fn extract_compact_pt<BE: Backend>(
    module: &Module<BE>,
    pt: &mut CKKSPlaintext<impl DataMut>,
    target_k: TorusPrecision,
    full: &GLWEPlaintext<impl DataRef>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: VecZnxNormalize<BE>,
{
    assert_eq!(full.base2k.0, pt.inner.base2k.0, "extract_compact_pt: base2k mismatch");
    let offset_bits = target_k.0 as i64 - (pt.inner.data.size() * pt.inner.base2k.0 as usize) as i64;
    if offset_bits % pt.inner.base2k.0 as i64 == 0 {
        let offset_limbs = (offset_bits / pt.inner.base2k.0 as i64) as usize;
        assert!(
            offset_limbs + pt.inner.data.size() <= full.data.size(),
            "extract_compact_pt: offset out of range"
        );
        for l in 0..pt.inner.data.size() {
            let src_limb: &[i64] = full.data.at(0, offset_limbs + l);
            let dst_limb: &mut [i64] = pt.inner.data.at_mut(0, l);
            dst_limb.copy_from_slice(src_limb);
        }
        return;
    }

    module.vec_znx_normalize(
        &mut pt.inner.data,
        pt.inner.base2k.0 as usize,
        offset_bits,
        0,
        &full.data,
        full.base2k.0 as usize,
        0,
        scratch,
    );
}

/// Carves an expanded [`GLWEPlaintext`] from scratch and fills it from a compact
/// [`CKKSPlaintext`]. Returns the expanded plaintext and remaining scratch.
pub(crate) fn offset_pt_from_scratch<'a, BE: Backend>(
    module: &Module<BE>,
    ct_n: Degree,
    ct_base2k: Base2K,
    full_k: TorusPrecision,
    target_k: TorusPrecision,
    pt: &CKKSPlaintext<impl DataRef>,
    scratch: &'a mut Scratch<BE>,
) -> (GLWEPlaintext<&'a mut [u8]>, &'a mut Scratch<BE>)
where
    Module<BE>: VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let layout = GLWEPlaintextLayout {
        n: ct_n,
        base2k: ct_base2k,
        k: full_k,
    };
    let (mut full, scratch_rest) = scratch.take_glwe_plaintext(&layout);
    fill_offset_pt(module, &mut full, target_k, pt, scratch_rest);
    (full, scratch_rest)
}

/// Returns the scratch bytes needed for [`offset_pt_from_scratch`].
pub(crate) fn offset_pt_scratch_bytes<BE: Backend>(
    module: &Module<BE>,
    ct_n: Degree,
    ct_base2k: Base2K,
    ct_k: TorusPrecision,
) -> usize
where
    Module<BE>: VecZnxNormalizeTmpBytes,
{
    GLWEPlaintext::bytes_of(ct_n, ct_base2k, ct_k) + module.vec_znx_normalize_tmp_bytes()
}

/// Carves a compact [`CKKSPlaintext`] from scratch and encodes a complex
/// constant into it.
///
/// This exists separately from `CKKSPlaintextPrepared::from_const` so the
/// non-prepared `*_const` leveled operations can stay scratch-based and avoid
/// heap allocation on the arithmetic path.
pub(crate) fn const_pt_from_scratch<'a, BE: Backend>(
    ct: &CKKSCiphertext<impl DataRef>,
    re: f64,
    im: f64,
    scratch: &'a mut Scratch<BE>,
) -> (CKKSPlaintext<&'a mut [u8]>, &'a mut Scratch<BE>)
where
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = ct.inner.n();
    let delta = (2.0f64).powi(ct.torus_scale_bits() as i32);
    let v_re = (delta * re).round() as i64;
    let v_im = (delta * im).round() as i64;

    let base2k = ct.inner.base2k();
    let embed_bits = ct.torus_scale_bits();
    let pt_k = TorusPrecision((embed_bits as usize).div_ceil(base2k.0 as usize) as u32 * base2k.0);

    let layout = GLWEPlaintextLayout { n, base2k, k: pt_k };
    let (mut inner, scratch_rest) = scratch.take_glwe_plaintext(&layout);

    // Zero all coefficients (scratch memory may contain garbage).
    inner.data.raw_mut().fill(0);

    inner.encode_coeff_i64(v_re, pt_k, 0);
    inner.encode_coeff_i64(v_im, pt_k, n.0 as usize / 2);

    (CKKSPlaintext { inner, embed_bits }, scratch_rest)
}

/// Returns the scratch bytes needed for [`const_pt_from_scratch`].
pub(crate) fn const_pt_scratch_bytes(ct: &CKKSCiphertext<impl Data>) -> usize {
    let base2k = ct.inner.base2k();
    let pt_k = TorusPrecision((ct.torus_scale_bits() as usize).div_ceil(base2k.0 as usize) as u32 * base2k.0);
    GLWEPlaintext::bytes_of(ct.inner.n(), base2k, pt_k)
}

/// Sign-extends the MSB limb of each column when `k` is not limb-aligned.
///
/// If `k % base2k != 0`, the top limb is only partially active.  The unused
/// upper bits must carry the sign bit so that Horner reconstruction interprets
/// the torus value correctly.
pub(crate) fn sign_extend_msb(ct: &mut CKKSCiphertext<impl DataMut>) {
    let base2k = ct.inner.base2k().0;
    let top = ct.inner.k().0 % base2k;
    if top != 0 {
        let shift = 64 - top as usize;
        let data = ct.inner.data_mut();
        for col in 0..data.cols() {
            let limb0: &mut [i64] = data.at_mut(col, 0);
            for v in limb0.iter_mut() {
                *v = (*v << shift) >> shift;
            }
        }
    }
}
