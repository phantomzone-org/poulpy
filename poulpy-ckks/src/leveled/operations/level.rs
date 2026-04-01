//! Level management: drop limbs or bits, rescale.

use crate::layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext};
use poulpy_core::{
    GLWEShift, ScratchTakeCore,
    layouts::{LWEInfos, SetGLWEInfos, TorusPrecision},
};
use poulpy_hal::layouts::{Backend, DataMut, Module, Scratch};

/// Drops `n` least-significant limbs from a plaintext.
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

/// Drops `b` bits of precision from a plaintext.
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

/// Drops `n` least-significant limbs from a ciphertext.
pub fn drop_limbs(ct: &mut CKKSCiphertext<impl DataMut>, n: usize) {
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

/// Drops `b` bits of precision from a ciphertext.
pub fn drop_bits(ct: &mut CKKSCiphertext<impl DataMut>, b: u32) {
    let base2k = ct.inner.base2k().0;
    let k = ct.inner.k();
    assert!(b <= k.0, "cannot drop {b} bits: exceeds precision k={}", k.0);
    ct.inner.data_mut().size -= (b / base2k) as usize;
    ct.inner.set_k(TorusPrecision(k.0 - b));
}

/// Returns scratch bytes needed by [`rescale`].
pub fn rescale_tmp_bytes<BE: Backend>(module: &Module<BE>) -> usize
where
    Module<BE>: GLWEShift<BE>,
{
    module.glwe_rsh_tmp_byte()
}

/// Rescales a ciphertext by `bits` bits, reducing `log_delta` accordingly.
pub fn rescale<BE: Backend>(module: &Module<BE>, ct: &mut CKKSCiphertext<impl DataMut>, bits: u32, scratch: &mut Scratch<BE>)
where
    Module<BE>: GLWEShift<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    assert!(bits <= ct.log_delta, "rescale: bits ({bits}) > log_delta ({})", ct.log_delta);
    module.glwe_rsh(bits as usize, &mut ct.inner, scratch);
    drop_bits(ct, bits);
    ct.log_delta -= bits;
}
