use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyPrepared, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    CKKSInfos, checked_log_budget_sub,
    layouts::{CKKSCiphertext, ciphertext::CKKSOffset},
};

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
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        key: &GLWEAutomorphismKeyPrepared<K, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Src: Data,
        K: Data,
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE>,
        GLWEAutomorphismKeyPrepared<K, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
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

        dst.meta = src.meta();
        dst.meta.log_budget = checked_log_budget_sub("conjugate", dst.log_budget(), offset)?;
        Ok(())
    }

    fn ckks_conjugate_assign_default<'s, Dst, K>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        key: &GLWEAutomorphismKeyPrepared<K, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        K: Data,
        Self: GLWEAutomorphism<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        GLWEAutomorphismKeyPrepared<K, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        let mut dst_ref = GLWEToBackendMut::<BE>::to_backend_mut(dst);
        self.glwe_automorphism_assign(&mut dst_ref, key, scratch);
        Ok(())
    }
}

impl<BE: Backend> CKKSConjugateDefault<BE> for Module<BE> {}
