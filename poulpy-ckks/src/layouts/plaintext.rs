use poulpy_core::layouts::{Base2K, Degree, GLWEPlaintext, GLWEPlaintextToMut, GLWEPlaintextToRef, TorusPrecision};
use poulpy_hal::layouts::{Data, DataMut, DataRef};

pub struct CKKSPlaintext<D: Data> {
    pub inner: GLWEPlaintext<D>,
    pub log_delta: u32,
}

impl CKKSPlaintext<Vec<u8>> {
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, log_delta: u32) -> Self {
        Self {
            inner: GLWEPlaintext::alloc(n, base2k, k),
            log_delta,
        }
    }
}

pub trait CKKSPlaintextToRef {
    fn to_ref(&self) -> CKKSPlaintext<&[u8]>;
}

impl<D: DataRef> CKKSPlaintextToRef for CKKSPlaintext<D> {
    fn to_ref(&self) -> CKKSPlaintext<&[u8]> {
        CKKSPlaintext {
            inner: self.inner.to_ref(),
            log_delta: self.log_delta,
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
            log_delta: self.log_delta,
        }
    }
}
