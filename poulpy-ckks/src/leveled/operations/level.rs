//! Level management, rescaling, and controlled precision drops.

use crate::layouts::ciphertext::CKKSCiphertext;
use poulpy_core::{
    GLWECopy, GLWEShift, ScratchTakeCore,
    layouts::{LWEInfos, TorusPrecision},
};
use poulpy_hal::layouts::{Backend, DataMut, Module, Scratch, ZnxInfos, ZnxViewMut};

/// Returns the scratch bytes needed for [`rescale`] (currently zero).
pub fn rescale_tmp_bytes<BE: Backend>(module: &Module<BE>) -> usize {
    let _ = module;
    0
}

/// Returns the scratch bytes needed for [`div_pow2`] and [`div_pow2_inplace`].
pub fn div_pow2_tmp_bytes<BE: Backend>(module: &Module<BE>) -> usize
where
    Module<BE>: GLWEShift<BE>,
{
    module.glwe_rsh_tmp_byte()
}

/// Rescales by consuming `bits` of active torus precision and reducing the
/// torus scale by `bits`.
///
/// In the bivariate representation this lowers `k`, `offset_bits`, and
/// `torus_scale_bits` together. The decoded message stays approximately
/// unchanged, while the consumed precision becomes visible in the active
/// ciphertext shape.
///
/// # Panics
///
/// Panics if `bits > torus_scale_bits`.
pub fn rescale<BE: Backend>(module: &Module<BE>, ct: &mut CKKSCiphertext<impl DataMut>, bits: u32, scratch: &mut Scratch<BE>)
where
    Scratch<BE>: ScratchTakeCore<BE>,
{
    ct.assert_valid("rescale input");
    assert!(
        bits <= ct.torus_scale_bits(),
        "rescale: bits ({bits}) > torus_scale_bits ({})",
        ct.torus_scale_bits()
    );
    assert!(
        bits <= ct.prefix_bits(),
        "rescale: bits ({bits}) > prefix_bits ({})",
        ct.prefix_bits()
    );

    let _ = module;
    let _ = scratch;
    let target_k = TorusPrecision(ct.prefix_bits() - bits);
    ct.set_active_k(target_k);
    ct.zero_inactive_tail();
    ct.offset_bits -= bits;
    ct.torus_scale_bits -= bits;
    super::utils::sign_extend_msb(ct);
    ct.assert_valid("rescale result");
}

/// Divides the encoded message by `2^bits` without changing CKKS metadata.
///
/// This is a scale-preserving arithmetic right shift on the torus payload:
/// `k`, `offset_bits`, `torus_scale_bits`, and `size` stay unchanged. The result therefore decodes
/// to approximately `message / 2^bits`, with truncation of discarded low bits.
pub fn div_pow2<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
    bits: usize,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEShift<BE> + GLWECopy,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    ct.assert_valid("div_pow2 input");
    module.glwe_copy(&mut res.inner, &ct.inner);
    res.torus_scale_bits = ct.torus_scale_bits();
    res.offset_bits = ct.offset_bits();
    module.glwe_rsh(bits, &mut res.inner, scratch);
    super::utils::sign_extend_msb(res);
    res.assert_valid("div_pow2 result");
}

/// Divides the encoded message by `2^bits` in place without changing CKKS metadata.
pub fn div_pow2_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    bits: usize,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEShift<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    ct.assert_valid("div_pow2_inplace input");
    module.glwe_rsh(bits, &mut ct.inner, scratch);
    super::utils::sign_extend_msb(ct);
    ct.assert_valid("div_pow2_inplace result");
}

/// Divides the encoded message by `2^bits` and drops the same number of scale bits.
/// The underlying message stays the same, and the scaling factor is reduced.
///
/// # Panics
///
/// Panics if `bits >= torus_scale_bits`, so the resulting `torus_scale_bits` stays strictly positive.
pub fn drop_scaling_precision<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
    bits: usize,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEShift<BE> + GLWECopy,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    ct.assert_valid("drop_scaling_precision input");
    assert!(
        (bits as u32) < ct.torus_scale_bits(),
        "drop_scaling_precision: bits ({bits}) must be < torus_scale_bits ({})",
        ct.torus_scale_bits()
    );
    div_pow2(module, res, ct, bits, scratch);
    res.torus_scale_bits -= bits as u32;
    res.assert_valid("drop_scaling_precision result");
}

/// Divides the encoded message by `2^bits` in place and drops the same number of scale bits.
///
/// This is the in-place form of [`drop_scaling_precision`]. `k` and `size` are
/// preserved, while `torus_scale_bits` is reduced by `bits` and kept strictly positive.
///
/// # Panics
///
/// Panics if `bits >= torus_scale_bits`.
pub fn drop_scaling_precision_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    bits: usize,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEShift<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    ct.assert_valid("drop_scaling_precision_inplace input");
    assert!(
        (bits as u32) < ct.torus_scale_bits(),
        "drop_scaling_precision_inplace: bits ({bits}) must be < torus_scale_bits ({})",
        ct.torus_scale_bits()
    );
    div_pow2_inplace(module, ct, bits, scratch);
    ct.torus_scale_bits -= bits as u32;
    ct.assert_valid("drop_scaling_precision_inplace result");
}

/// Reduces the torus precision of a ciphertext to `target_k`
/// without changing `torus_scale_bits`.
///
/// This is a prefix truncation of the bivariate representation: the ciphertext
/// keeps the same encoded message, but both the active torus window and the
/// message position are lowered to the retained precision.
pub fn drop_torus_precision(ct: &mut CKKSCiphertext<impl DataMut>, target_k: TorusPrecision) {
    ct.assert_valid("drop_torus_precision input");
    assert!(
        target_k.0 <= ct.prefix_bits(),
        "drop_torus_precision: target_k ({}) > current k ({})",
        target_k.0,
        ct.prefix_bits()
    );
    assert!(
        target_k.0 >= ct.torus_scale_bits(),
        "drop_torus_precision: target_k ({}) < torus_scale_bits ({})",
        target_k.0,
        ct.torus_scale_bits()
    );
    let base2k = ct.inner.base2k().0;
    let dropped_bits = ct.prefix_bits() - target_k.0;
    assert_eq!(
        dropped_bits % base2k,
        0,
        "drop_torus_precision: only limb-aligned truncation is supported (dropped_bits={dropped_bits}, base2k={base2k})"
    );
    let dropped_limbs = dropped_bits.div_ceil(base2k) as usize;
    if dropped_limbs > 0 {
        let data = ct.inner.data_mut();
        let limb_stride = data.n() * data.cols();
        let src_start = dropped_limbs * limb_stride;
        let src_end = src_start + (data.size - dropped_limbs) * limb_stride;
        data.raw_mut().copy_within(src_start..src_end, 0);
    }
    ct.set_active_k(target_k);
    ct.zero_inactive_tail();
    ct.offset_bits = ct.offset_bits().min(target_k.0);
    super::utils::sign_extend_msb(ct);
    ct.assert_valid("drop_torus_precision result");
}
