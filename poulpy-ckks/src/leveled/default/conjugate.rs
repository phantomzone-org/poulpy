use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCiphertextToBackendMut, CKKSCiphertextToBackendRef, CKKSInfos, SetCKKSInfos, checked_log_budget_sub, layouts::ciphertext::CKKSOffset};

pub(crate) trait CKKSConjugateDefault<BE: Backend> {
    fn ckks_conjugate_tmp_bytes_default<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        self.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
    }

    fn ckks_conjugate_into_default<'s, Dst, Src, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSOffset + CKKSInfos + SetCKKSInfos,
        Src: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        let offset = dst.offset_unary(src);
        if offset != 0 {
            self.glwe_lsh(dst, src, offset, scratch);
            let mut dst_ref = GLWEToBackendMut::<BE>::to_backend_mut(dst);
            self.glwe_automorphism_assign(&mut dst_ref, key, scratch);
        } else {
            let mut dst_ref = GLWEToBackendMut::<BE>::to_backend_mut(dst);
            let src_ref = GLWEToBackendRef::<BE>::to_backend_ref(src);
            self.glwe_automorphism(&mut dst_ref, &src_ref, key, scratch);
        }

        dst.set_meta(src.meta());
        dst.set_log_budget(checked_log_budget_sub("conjugate", dst.log_budget(), offset)?);
        Ok(())
    }

    fn ckks_conjugate_assign_default<'s, Dst, K>(
        &self,
        dst: &mut Dst,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE>,
        Dst: CKKSCiphertextToBackendMut<BE>,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        let mut dst_ref = GLWEToBackendMut::<BE>::to_backend_mut(dst);
        self.glwe_automorphism_assign(&mut dst_ref, key, scratch);
        Ok(())
    }
}

impl<BE: Backend> CKKSConjugateDefault<BE> for Module<BE> {}
