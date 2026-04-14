//! CKKS ciphertext slot rotation.
//!
//! Slot rotation is an automorphism whose Galois element is determined by
//! the rotation amount. The public API is expressed in terms of the slot
//! shift `k`, and the provided key store is indexed by that same shift.

use crate::{CKKS, CKKSInfos};
use poulpy_core::{
    GLWEAutomorphism, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToRef, GetGaloisElement},
};
use poulpy_hal::layouts::{Backend, DataMut, Module, Scratch};

pub trait CKKSRotateOps {
    fn rotate_tmp_bytes<BE: Backend, C, K>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Module<BE>: GLWEAutomorphism<BE>;

    /// `self = Rotate(ct, k)` — rotates slots by `k` positions.
    fn rotate<O, H, K, BE: Backend>(&mut self, module: &Module<BE>, other: &O, k: i64, keys: &H, scratch: &mut Scratch<BE>)
    where
        Module<BE>: GLWEAutomorphism<BE>,
        O: GLWEToRef + GLWEInfos + CKKSInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    /// `self = Rotate(self, k)` — rotates slots by `k` positions in place.
    fn rotate_inplace<H, K, BE: Backend>(&mut self, module: &Module<BE>, k: i64, keys: &H, scratch: &mut Scratch<BE>)
    where
        Module<BE>: GLWEAutomorphism<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<D: DataMut> CKKSRotateOps for GLWE<D, CKKS> {
    fn rotate_tmp_bytes<BE: Backend, C, K>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Module<BE>: GLWEAutomorphism<BE>,
    {
        module.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
    }

    /// `self = Rotate(ct, k)` — rotates slots by `k` positions.
    fn rotate<O, H, K, BE: Backend>(&mut self, module: &Module<BE>, other: &O, k: i64, keys: &H, scratch: &mut Scratch<BE>)
    where
        Module<BE>: GLWEAutomorphism<BE>,
        O: GLWEToRef + GLWEInfos + CKKSInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        // TODO: manage case where receiver has smaller k
        let key = keys
            .get_automorphism_key(k)
            .unwrap_or_else(|| panic!("missing automorphism key for rotation {k}"));
        module.glwe_automorphism(self, other, key, scratch);
        self.meta = other.meta();
    }

    /// `self = Rotate(self, k)` — rotates slots by `k` positions in place.
    fn rotate_inplace<H, K, BE: Backend>(&mut self, module: &Module<BE>, k: i64, keys: &H, scratch: &mut Scratch<BE>)
    where
        Module<BE>: GLWEAutomorphism<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let key = keys
            .get_automorphism_key(k)
            .unwrap_or_else(|| panic!("missing automorphism key for rotation {k}"));
        module.glwe_automorphism_inplace(self, key, scratch);
    }
}
