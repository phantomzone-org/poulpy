//! CKKS ciphertext complex conjugation.
//!
//! Complex conjugation is the automorphism for Galois element `-1`.
//! The caller must supply the prepared automorphism key for that element.

use crate::layouts::ciphertext::CKKSCiphertext;
use poulpy_core::{
    GLWEAutomorphism, ScratchTakeCore,
    layouts::{GGLWEInfos, GLWEAutomorphismKeyPrepared, GLWEInfos},
};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

impl CKKSCiphertext<Vec<u8>> {
    /// Returns the scratch bytes needed for conjugate / rotate operations.
    pub fn automorphism_tmp_bytes<C, K, BE: Backend>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Module<BE>: GLWEAutomorphism<BE>,
    {
        module.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
    }
}

impl<D: DataMut> CKKSCiphertext<D> {
    /// `self = Conjugate(ct)` using the conjugation key (Galois element -1).
    pub fn conjugate<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        ct: &CKKSCiphertext<impl DataRef>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: GLWEAutomorphism<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism(&mut self.inner, &ct.inner, key, scratch);
        self.log_delta = ct.log_delta;
    }

    /// `self = Conjugate(self)` using the conjugation key (Galois element -1).
    pub fn conjugate_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: GLWEAutomorphism<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_inplace(&mut self.inner, key, scratch);
    }
}
