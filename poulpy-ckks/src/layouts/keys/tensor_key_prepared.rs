//! Prepared (DFT-domain) CKKS relinearization key.

use super::tensor_key::CKKSTensorKey;
use poulpy_core::layouts::{Base2K, Degree, GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory, LWEInfos, TorusPrecision};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, DataRef, Module, Scratch},
};

/// Prepared relinearization key for fast tensor relinearization.
pub struct CKKSTensorKeyPrepared<D: Data, BE: Backend> {
    pub inner: GLWETensorKeyPrepared<D, BE>,
}

impl<BE: Backend> CKKSTensorKeyPrepared<Vec<u8>, BE> {
    /// Allocates a prepared tensor key from CKKS parameters.
    pub fn alloc(module: &Module<BE>, n: Degree, base2k: Base2K, k: TorusPrecision) -> Self
    where
        Module<BE>: GLWETensorKeyPreparedFactory<BE>,
    {
        CKKSTensorKeyPrepared {
            inner: GLWETensorKeyPrepared::alloc_from_infos(module, &CKKSTensorKey::layout(n, base2k, k)),
        }
    }

    /// Prepares the tensor key from its unprepared form.
    pub fn prepare(&mut self, module: &Module<BE>, tsk: &CKKSTensorKey<impl DataRef>, scratch: &mut Scratch<BE>)
    where
        Module<BE>: GLWETensorKeyPreparedFactory<BE>,
        Scratch<BE>: ScratchAvailable,
    {
        self.inner.prepare(module, &tsk.inner, scratch);
    }
}

impl<D: Data + DataRef, BE: Backend> CKKSTensorKeyPrepared<D, BE> {
    pub fn size(&self) -> usize {
        self.inner.size()
    }
}
