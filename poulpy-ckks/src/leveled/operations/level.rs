//! Level management and scale-preserving division by powers of two.

use crate::layouts::ciphertext::CKKSCiphertext;
use poulpy_core::{GLWECopy, GLWEShift, ScratchTakeCore, layouts::{LWEInfos, SetGLWEInfos, TorusPrecision}};
use poulpy_hal::layouts::{Backend, DataMut, Module, Scratch};

fn assert_target_k(label: &str, current_k: u32, target_k: TorusPrecision) {
    assert!(
        target_k.0 > 0,
        "{label}: target k must be strictly positive, got {}",
        target_k.0
    );
    assert!(
        target_k.0 <= current_k,
        "{label}: target k ({}) exceeds current k ({current_k})",
        target_k.0
    );
}

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

/// Rescales by shrinking the active precision window by `bits`.
///
/// After rescale: `new_k = k - bits`, `new_log_delta = log_delta - bits`,
/// and `new_size = ceil(new_k / base2k)`.  The MSB limb is sign-extended
/// if `new_k` is not limb-aligned.
///
/// # Panics
///
/// Panics if `bits > log_delta` or if the resulting `k` would be zero.
pub fn rescale<BE: Backend>(module: &Module<BE>, ct: &mut CKKSCiphertext<impl DataMut>, bits: u32, scratch: &mut Scratch<BE>) {
    let _ = module;
    let _ = scratch;
    assert!(bits <= ct.log_delta, "rescale: bits ({bits}) > log_delta ({})", ct.log_delta);
    assert!(
        ct.inner.k().0 > bits,
        "rescale: k ({}) exhausted (need > {bits})",
        ct.inner.k().0
    );
    let new_k = TorusPrecision(ct.inner.k().0 - bits);
    assert_target_k("rescale", ct.inner.k().0, new_k);
    let base2k = ct.inner.base2k().0;
    ct.inner.set_k(new_k);
    ct.inner.data_mut().size = new_k.0.div_ceil(base2k) as usize;
    super::utils::sign_extend_msb(ct);
    ct.log_delta -= bits;
}

/// Divides the encoded message by `2^bits` without changing CKKS metadata.
///
/// This is a scale-preserving arithmetic right shift on the torus payload:
/// `k`, `log_delta`, and `size` stay unchanged. The result therefore decodes
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
    module.glwe_copy(&mut res.inner, &ct.inner);
    res.log_delta = ct.log_delta;
    module.glwe_rsh(bits, &mut res.inner, scratch);
    super::utils::sign_extend_msb(res);
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
    module.glwe_rsh(bits, &mut ct.inner, scratch);
    super::utils::sign_extend_msb(ct);
}
