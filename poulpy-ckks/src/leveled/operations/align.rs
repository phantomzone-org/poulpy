//! Ciphertext alignment and precision-window helpers.
//!
//! Two ciphertexts are *aligned* when they share the same `offset_bits` and
//! `prefix_bits`. Add/sub require aligned operands; [`common_window`]
//! computes the intersection window and [`align_to`] / [`align_to_inplace`]
//! re-window a ciphertext to match.

use crate::layouts::ciphertext::{CKKSCiphertext, CKKSCiphertextToRef};
use poulpy_core::{
    GLWEAlign, ScratchTakeCore,
    layouts::{GLWE, GLWEInfos, GLWELayout, LWEInfos, TorusPrecision},
};
use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut};

/// Returns `true` if both ciphertexts share the same `offset_bits` and `prefix_bits`.
pub fn are_cts_aligned(a: &CKKSCiphertext<impl Data>, b: &CKKSCiphertext<impl Data>) -> bool {
    a.offset_bits() == b.offset_bits() && a.prefix_bits() == b.prefix_bits()
}

/// Panics if the two ciphertexts are not aligned (same `offset_bits` and `prefix_bits`).
pub fn assert_cts_aligned(a: &CKKSCiphertext<impl Data>, b: &CKKSCiphertext<impl Data>, label: &str) {
    assert!(
        are_cts_aligned(a, b),
        "{label}: ciphertexts are not aligned (lhs: offset_bits={}, prefix_bits={}; rhs: offset_bits={}, prefix_bits={})",
        a.offset_bits(),
        a.prefix_bits(),
        b.offset_bits(),
        b.prefix_bits()
    );
}

/// Computes the common precision window of two ciphertexts.
///
/// Returns `(offset_common, target_k)` where `offset_common` is the larger
/// of the two offsets and `target_k` is `offset_common + min(payload_a, payload_b)`.
pub fn common_window(a: &CKKSCiphertext<impl Data>, b: &CKKSCiphertext<impl Data>) -> (u32, TorusPrecision) {
    let payload_a = a.prefix_bits() - a.offset_bits();
    let payload_b = b.prefix_bits() - b.offset_bits();
    let offset_common = a.offset_bits().max(b.offset_bits());
    let payload_common = payload_a.min(payload_b);
    (offset_common, TorusPrecision(offset_common + payload_common))
}

/// Returns the scratch bytes needed for [`align_to`].
pub fn align_to_tmp_bytes<BE: Backend>(module: &Module<BE>, src: &CKKSCiphertext<impl Data>, target_k: TorusPrecision) -> usize
where
    Module<BE>: GLWEAlign<BE>,
{
    let layout = GLWELayout {
        n: src.inner.n(),
        base2k: src.inner.base2k(),
        k: target_k,
        rank: src.inner.rank(),
    };
    module.glwe_align_tmp_bytes(&layout, &src.inner)
}

/// Re-windows `src` into `res` at the given `target_offset_bits` and `target_k`.
pub fn align_to<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    src: &CKKSCiphertext<impl DataRef>,
    target_offset_bits: u32,
    target_k: TorusPrecision,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEAlign<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    src.assert_valid("align_to source");
    assert_eq!(res.inner.base2k(), src.inner.base2k(), "align_to: base2k mismatch");
    assert_eq!(res.inner.rank(), src.inner.rank(), "align_to: rank mismatch");
    assert!(
        target_offset_bits <= target_k.0,
        "align_to: target_offset_bits ({target_offset_bits}) exceeds target_k ({})",
        target_k.0
    );
    assert!(
        src.torus_scale_bits() <= target_offset_bits,
        "align_to: target_offset_bits ({target_offset_bits}) is below torus_scale_bits ({})",
        src.torus_scale_bits()
    );

    res.set_active_k(target_k);
    res.zero_inactive_tail();
    module.glwe_align(&mut res.inner, target_offset_bits, &src.inner, src.offset_bits(), scratch);
    res.offset_bits = target_offset_bits;
    res.torus_scale_bits = src.torus_scale_bits();
    res.assert_valid("align_to result");
}

/// Returns the scratch bytes needed for [`align_to_inplace`].
pub fn align_to_inplace_tmp_bytes<BE: Backend>(
    module: &Module<BE>,
    ct: &CKKSCiphertext<impl Data>,
    target_k: TorusPrecision,
) -> usize
where
    Module<BE>: GLWEAlign<BE>,
{
    let layout = GLWELayout {
        n: ct.inner.n(),
        base2k: ct.inner.base2k(),
        k: target_k,
        rank: ct.inner.rank(),
    };
    GLWE::bytes_of_from_infos(&layout) + module.glwe_align_tmp_bytes(&layout, &ct.inner)
}

/// Re-windows `ct` in place to the given `target_offset_bits` and `target_k`.
pub fn align_to_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    target_offset_bits: u32,
    target_k: TorusPrecision,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEAlign<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    ct.assert_valid("align_to_inplace input");
    if ct.offset_bits() == target_offset_bits && ct.prefix_bits() == target_k.0 {
        return;
    }

    let layout = GLWELayout {
        n: ct.inner.n(),
        base2k: ct.inner.base2k(),
        k: target_k,
        rank: ct.inner.rank(),
    };
    let (tmp_inner, scratch_rest) = scratch.take_glwe(&layout);
    let mut tmp = CKKSCiphertext {
        inner: tmp_inner,
        offset_bits: target_offset_bits,
        torus_scale_bits: ct.torus_scale_bits(),
    };
    align_to(module, &mut tmp, &ct.to_ref(), target_offset_bits, target_k, scratch_rest);

    ct.set_active_k(target_k);
    ct.zero_inactive_tail();
    let raw = tmp.inner.data().raw();
    ct.inner.data_mut().raw_mut()[..raw.len()].copy_from_slice(raw);
    ct.offset_bits = target_offset_bits;
    ct.assert_valid("align_to_inplace result");
}
