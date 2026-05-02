use anyhow::Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, prepared::GLWEAutomorphismKeyPreparedToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCiphertextToBackendMut, CKKSCiphertextToBackendRef};

use crate::{CKKSInfos, SetCKKSInfos, oep::CKKSImpl};

pub trait CKKSConjugateOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_conjugate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos + CKKSInfos,
        K: GGLWEInfos;

    fn ckks_conjugate_into<Dst, Src, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEInfos;

    fn ckks_conjugate_assign<Dst, K>(&self, dst: &mut Dst, key: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEInfos;
}
