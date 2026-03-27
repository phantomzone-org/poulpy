//! Level management: drop limbs or bits from ciphertexts and plaintexts.
//!
//! In CKKS, the "level" of a ciphertext tracks its remaining precision. Each
//! multiplication consumes `log_delta` bits of precision (not necessarily a
//! whole limb). These functions reduce the level by discarding least-significant
//! limbs or an arbitrary number of bits, updating only the metadata fields
//! (`size`, `k`). No data movement or zero-fill is performed; the dropped limb
//! data remains in the buffer but is no longer considered active.

use crate::layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext};
use poulpy_core::layouts::{LWEInfos, SetGLWEInfos, TorusPrecision};
use poulpy_hal::layouts::DataMut;

// --- Plaintext ----------------------------------------------------------------

/// Drops `n` least-significant limbs from a CKKS plaintext.
///
/// Each limb is `base2k` bits wide, so this removes `n * base2k` bits of
/// precision. Panics if `n` limbs would reduce `k` below zero.
pub fn drop_limbs_pt(pt: &mut CKKSPlaintext<impl DataMut>, n: usize) {
    let n_bits = n as u32 * pt.inner.base2k.0;
    assert!(
        n_bits <= pt.inner.k.0,
        "cannot drop {n} limbs ({n_bits} bits): exceeds precision k={}",
        pt.inner.k.0
    );
    pt.inner.data.size -= n;
    pt.inner.k.0 -= n_bits;
}

/// Drops `b` bits of least-significant precision from a CKKS plaintext.
///
/// Removes `b / base2k` complete limbs and reduces `k` by `b` bits in total.
/// If `b` is not a multiple of `base2k`, the last active limb retains the
/// residual bits without zeroing; only `k` is adjusted. Panics if `b > k`.
pub fn drop_bits_pt(pt: &mut CKKSPlaintext<impl DataMut>, b: u32) {
    assert!(
        b <= pt.inner.k.0,
        "cannot drop {b} bits: exceeds precision k={}",
        pt.inner.k.0
    );
    let base2k = pt.inner.base2k.0;
    pt.inner.data.size -= (b / base2k) as usize;
    pt.inner.k.0 -= b;
}

// --- Ciphertext ---------------------------------------------------------------

/// Drops `n` least-significant limbs from a CKKS ciphertext.
///
/// Each limb is `base2k` bits wide, so this removes `n * base2k` bits of
/// precision. Panics if `n` limbs would reduce `k` below zero.
pub fn drop_limbs_ct(ct: &mut CKKSCiphertext<impl DataMut>, n: usize) {
    let n_bits = n as u32 * ct.inner.base2k().0;
    let k = ct.inner.k();
    assert!(
        n_bits <= k.0,
        "cannot drop {n} limbs ({n_bits} bits): exceeds precision k={}",
        k.0
    );
    ct.inner.data_mut().size -= n;
    ct.inner.set_k(TorusPrecision(k.0 - n_bits));
}

/// Drops `b` bits of least-significant precision from a CKKS ciphertext.
///
/// Removes `b / base2k` complete limbs and reduces `k` by `b` bits in total.
/// If `b` is not a multiple of `base2k`, the last active limb retains the
/// residual bits without zeroing; only `k` is adjusted. Panics if `b > k`.
pub fn drop_bits_ct(ct: &mut CKKSCiphertext<impl DataMut>, b: u32) {
    let base2k = ct.inner.base2k().0;
    let k = ct.inner.k();
    assert!(
        b <= k.0,
        "cannot drop {b} bits: exceeds precision k={}",
        k.0
    );
    ct.inner.data_mut().size -= (b / base2k) as usize;
    ct.inner.set_k(TorusPrecision(k.0 - b));
}
