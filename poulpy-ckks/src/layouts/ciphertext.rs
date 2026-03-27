use poulpy_core::layouts::{Base2K, Degree, GLWE, GLWELayout, GLWEToMut, GLWEToRef, Rank, TorusPrecision};
use poulpy_hal::layouts::{Data, DataMut, DataRef};

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
