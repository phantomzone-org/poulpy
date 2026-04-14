//! CKKS ciphertext complex conjugation.
//!
//! Complex conjugation is the automorphism for Galois element `-1`.
//! The caller must supply the prepared automorphism key for that element.

use poulpy_core::{
    GLWEAutomorphism, ScratchTakeCore,
    layouts::{GGLWEInfos, GLWE, GLWEAutomorphismKeyPrepared, GLWEInfos},
};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::CKKS;

pub trait CKKSConjugateOps {
    fn conjugate_tmp_bytes<C, K, BE: Backend>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Module<BE>: GLWEAutomorphism<BE>;

    /// `self = Conjugate(ct)` using the conjugation key (Galois element -1).
    fn conjugate<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        ct: &GLWE<impl DataRef, CKKS>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: GLWEAutomorphism<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    /// `self = Conjugate(self)` using the conjugation key (Galois element -1).
    fn conjugate_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: GLWEAutomorphism<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<D: DataMut> CKKSConjugateOps for GLWE<D, CKKS> {
    fn conjugate_tmp_bytes<C, K, BE: Backend>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Module<BE>: GLWEAutomorphism<BE>,
    {
        module.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
    }
    fn conjugate<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        ct: &GLWE<impl DataRef, CKKS>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: GLWEAutomorphism<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        // TODO: manage case where receiver has smaller k
        module.glwe_automorphism(self, ct, key, scratch);
        self.meta = ct.meta;
    }

    fn conjugate_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: GLWEAutomorphism<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_inplace(self, key, scratch);
    }
}
