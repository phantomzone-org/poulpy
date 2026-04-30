use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
        GetGaloisElement,
    },
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSInfos, checked_log_budget_sub,
    layouts::{CKKSCiphertext, ciphertext::CKKSOffset},
};

pub(crate) trait CKKSRotateDefault<BE: Backend> {
    fn ckks_rotate_tmp_bytes_default<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        self.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
    }

    fn ckks_rotate_into_default<'s, Dst, Src, H, K>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Src: Data,
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        let key = keys
            .get_automorphism_key(k)
            .ok_or(CKKSCompositionError::MissingAutomorphismKey {
                op: "rotate",
                rotation: k,
            })?;

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
        dst.meta.log_budget = checked_log_budget_sub("rotate", dst.log_budget(), offset)?;
        Ok(())
    }

    fn ckks_rotate_assign_default<'s, Dst, H, K>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Self: GLWEAutomorphism<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        let key = keys
            .get_automorphism_key(k)
            .ok_or(CKKSCompositionError::MissingAutomorphismKey {
                op: "rotate_assign",
                rotation: k,
            })?;
        let mut dst_ref = GLWEToBackendMut::<BE>::to_backend_mut(dst);
        self.glwe_automorphism_assign(&mut dst_ref, key, scratch);
        Ok(())
    }
}

impl<BE: Backend> CKKSRotateDefault<BE> for Module<BE> {}
