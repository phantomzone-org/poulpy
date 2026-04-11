//! CKKS ciphertext slot rotation.
//!
//! Slot rotation is an automorphism whose Galois element is determined by
//! the rotation amount. The public API is expressed in terms of the slot
//! shift `k`, and the provided key store is indexed by that same shift.

use crate::layouts::ciphertext::CKKSCiphertext;
use poulpy_core::{
    GLWEAutomorphism, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GLWEAutomorphismKeyHelper, GetGaloisElement},
};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

impl<D: DataMut> CKKSCiphertext<D> {
    /// `self = Rotate(ct, k)` — rotates slots by `k` positions.
    pub fn rotate<H, K, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        ct: &CKKSCiphertext<impl DataRef>,
        k: i64,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: GLWEAutomorphism<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        // TODO: manage case where receiver has smaller k
        let key = keys
            .get_automorphism_key(k)
            .unwrap_or_else(|| panic!("missing automorphism key for rotation {k}"));
        module.glwe_automorphism(&mut self.inner, &ct.inner, key, scratch);
        self.prec = ct.prec;
    }

    /// `self = Rotate(self, k)` — rotates slots by `k` positions in place.
    pub fn rotate_inplace<H, K, BE: Backend>(&mut self, module: &Module<BE>, k: i64, keys: &H, scratch: &mut Scratch<BE>)
    where
        Module<BE>: GLWEAutomorphism<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let key = keys
            .get_automorphism_key(k)
            .unwrap_or_else(|| panic!("missing automorphism key for rotation {k}"));
        module.glwe_automorphism_inplace(&mut self.inner, key, scratch);
    }
}
