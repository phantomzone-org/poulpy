use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GLWEAutomorphismKeyHelper, GLWEInfos, GetGaloisElement},
};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::{layouts::CKKSCiphertext, oep::CKKSImpl};

use super::{api::CKKSRotateOps, oep::CKKSRotateOep};

impl<BE: Backend + CKKSImpl<BE>> CKKSRotateOps<BE> for Module<BE> {
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        CKKSRotateOep::ckks_rotate_tmp_bytes(self, ct_infos, key_infos)
    }

    fn ckks_rotate<H, K>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        k: i64,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        CKKSRotateOep::ckks_rotate(self, dst, src, k, keys, scratch)
    }

    fn ckks_rotate_inplace<H, K>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        k: i64,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        CKKSRotateOep::ckks_rotate_inplace(self, dst, k, keys, scratch)
    }
}
