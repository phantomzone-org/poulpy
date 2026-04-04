//! Compact CKKS plaintext layout.
//!
//! A [`CKKSPlaintext`] stores the integer polynomial produced by CKKS encoding
//! using just enough limbs to hold `embed_bits`. It does not carry
//! ciphertext metadata such as `offset_bits`; torus placement into a ciphertext
//! window happens only when the plaintext participates in an operation
//! (encryption, add, sub, mul) or is prepared.

use poulpy_core::layouts::{Base2K, Degree, GLWEPlaintext, GLWEPlaintextToMut, GLWEPlaintextToRef, TorusPrecision};
use poulpy_hal::layouts::{Data, DataMut, DataRef};

/// Compact CKKS plaintext: an integer polynomial plus the embedding precision
/// `embed_bits`, stored with `ceil(embed_bits / base2k)` limbs.
///
/// This is a storage object.  It becomes an operation object when expanded
/// into a ciphertext-shaped torus layout by [`fill_offset_pt`] or by
/// preparing into a [`CKKSPlaintextPrepared`](super::plaintext_prepared::CKKSPlaintextPrepared).
///
/// ## Lifecycle
///
/// 1. Allocate with [`CKKSPlaintext::alloc`] (compact) or
///    [`CKKSPlaintext::alloc_for_decryption`] (full-width for decrypt output).
/// 2. Encode with [`encode`](crate::encoding::classical::encode).
/// 3. Use in encryption or leveled operations.
/// 4. After decryption, decode with [`decode`](crate::encoding::classical::decode).
///
/// [`fill_offset_pt`]: crate::leveled::operations::utils::fill_offset_pt
pub struct CKKSPlaintext<D: Data> {
    pub inner: GLWEPlaintext<D>,
    pub embed_bits: u32,
}

impl CKKSPlaintext<Vec<u8>> {
    /// Allocates a compact CKKS plaintext with just enough limbs for the
    /// message precision (`ceil(embed_bits / base2k)` limbs).
    pub fn alloc(n: Degree, base2k: Base2K, embed_bits: u32) -> Self {
        let pt_k = TorusPrecision((embed_bits as usize).div_ceil(base2k.0 as usize) as u32 * base2k.0);
        Self {
            inner: GLWEPlaintext::alloc(n, base2k, pt_k),
            embed_bits,
        }
    }

    /// Allocates a CKKS plaintext with the given Torus precision k.
    /// Used for decryption output, where the precision matches the ciphertext.
    pub fn alloc_for_decryption(n: Degree, base2k: Base2K, k: TorusPrecision, embed_bits: u32) -> Self {
        Self {
            inner: GLWEPlaintext::alloc(n, base2k, k),
            embed_bits,
        }
    }
}

impl<D: Data> CKKSPlaintext<D> {
    pub fn embed_bits(&self) -> u32 {
        self.embed_bits
    }
}

pub trait CKKSPlaintextToRef {
    fn to_ref(&self) -> CKKSPlaintext<&[u8]>;
}

impl<D: DataRef> CKKSPlaintextToRef for CKKSPlaintext<D> {
    fn to_ref(&self) -> CKKSPlaintext<&[u8]> {
        CKKSPlaintext {
            inner: self.inner.to_ref(),
            embed_bits: self.embed_bits,
        }
    }
}

pub trait CKKSPlaintextToMut {
    fn to_mut(&mut self) -> CKKSPlaintext<&mut [u8]>;
}

impl<D: DataMut> CKKSPlaintextToMut for CKKSPlaintext<D> {
    fn to_mut(&mut self) -> CKKSPlaintext<&mut [u8]> {
        CKKSPlaintext {
            inner: self.inner.to_mut(),
            embed_bits: self.embed_bits,
        }
    }
}
