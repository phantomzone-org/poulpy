//! CKKS ciphertext layout.
//!
//! A [`CKKSCiphertext`] pairs a rank-1 [`GLWE`] ciphertext with the scaling
//! factor `log_delta`.

use poulpy_core::layouts::{Base2K, Degree, GLWE, GLWELayout, GLWEToMut, GLWEToRef, Rank, TorusPrecision};
use poulpy_hal::layouts::{Data, DataMut, DataRef};

/// Encrypted CKKS value carrying a GLWE ciphertext and a scaling factor.
///
/// The active CKKS message lives in the top `k` bits of the torus window.
/// `log_delta` records the binary scaling factor applied during encoding:
/// the plaintext was multiplied by `2^{log_delta}` before encryption.
/// Each multiplication adds the operands' `log_delta` values; each rescale
/// subtracts the consumed bits from both `k` and `log_delta`.
///
/// ## Lifecycle
///
/// 1. Allocate with [`CKKSCiphertext::alloc`].
/// 2. Encrypt a compact [`CKKSPlaintext`](super::plaintext::CKKSPlaintext)
///    with [`encrypt_sk`](crate::leveled::encryption::encrypt_sk).
/// 3. Perform leveled arithmetic (add, sub, mul, rotate, conjugate).
/// 4. Decrypt with [`decrypt`](crate::leveled::encryption::decrypt).
///
/// ## Invariants
///
/// - `inner` is rank-1 (single GLWE polynomial pair).
/// - `inner.k()` is the active torus precision; `inner.base2k() * inner.size()`
///   is the physical width (may be larger than `k` after non-aligned rescale).
/// - `log_delta <= k` at all times.
pub struct CKKSCiphertext<D: Data> {
    pub inner: GLWE<D>,
    pub log_delta: u32,
}

impl CKKSCiphertext<Vec<u8>> {
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, log_delta: u32) -> Self {
        let infos = GLWELayout {
            n,
            base2k,
            k,
            rank: Rank(1),
        };
        Self {
            inner: GLWE::alloc_from_infos(&infos),
            log_delta,
        }
    }
}

pub trait CKKSCiphertextToRef {
    fn to_ref(&self) -> CKKSCiphertext<&[u8]>;
}

impl<D: DataRef> CKKSCiphertextToRef for CKKSCiphertext<D> {
    fn to_ref(&self) -> CKKSCiphertext<&[u8]> {
        CKKSCiphertext {
            inner: self.inner.to_ref(),
            log_delta: self.log_delta,
        }
    }
}

pub trait CKKSCiphertextToMut {
    fn to_mut(&mut self) -> CKKSCiphertext<&mut [u8]>;
}

impl<D: DataMut> CKKSCiphertextToMut for CKKSCiphertext<D> {
    fn to_mut(&mut self) -> CKKSCiphertext<&mut [u8]> {
        CKKSCiphertext {
            inner: self.inner.to_mut(),
            log_delta: self.log_delta,
        }
    }
}
