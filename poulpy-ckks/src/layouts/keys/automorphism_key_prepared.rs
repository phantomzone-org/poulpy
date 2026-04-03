//! Prepared (DFT-domain) CKKS automorphism key.

use super::automorphism_key::CKKSAutomorphismKey;
use poulpy_core::layouts::{Base2K, Degree, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory, TorusPrecision};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, DataRef, Module, Scratch},
};

/// DFT-domain automorphism key for hot-path slot rotations and conjugation.
///
/// Created by preparing a [`CKKSAutomorphismKey`] via
/// [`CKKSAutomorphismKeyPrepared::prepare`].
pub struct CKKSAutomorphismKeyPrepared<D: Data, BE: Backend> {
    pub inner: GLWEAutomorphismKeyPrepared<D, BE>,
}

impl<BE: Backend> CKKSAutomorphismKeyPrepared<Vec<u8>, BE> {
    /// Allocates a prepared automorphism key from CKKS parameters.
    pub fn alloc(module: &Module<BE>, n: Degree, base2k: Base2K, k: TorusPrecision) -> Self
    where
        Module<BE>: GLWEAutomorphismKeyPreparedFactory<BE>,
    {
        CKKSAutomorphismKeyPrepared {
            inner: GLWEAutomorphismKeyPrepared::alloc_from_infos(module, &CKKSAutomorphismKey::layout(n, base2k, k)),
        }
    }

    /// Prepares the automorphism key from its unprepared form.
    pub fn prepare(&mut self, module: &Module<BE>, atk: &CKKSAutomorphismKey<impl DataRef>, scratch: &mut Scratch<BE>)
    where
        Module<BE>: GLWEAutomorphismKeyPreparedFactory<BE>,
        Scratch<BE>: ScratchAvailable,
    {
        self.inner.prepare(module, &atk.inner, scratch);
    }
}
