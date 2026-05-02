use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWEAutomorphismKeyPreparedBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCiphertextToBackendMut, CKKSCiphertextToBackendRef};

use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub, ckks_offset_unary};

pub(crate) trait CKKSRotateDefault<BE: Backend> {
    fn ckks_rotate_tmp_bytes_default<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        self.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
    }

    fn ckks_rotate_into_default<'s, Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        Dst: CKKSCiphertextToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: CKKSCiphertextToBackendRef<BE> + LWEInfos + CKKSInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        let offset = ckks_offset_unary(dst, src);

        if offset != 0 {
            self.glwe_lsh(dst, src, offset, scratch);
            let mut dst_ref = GLWEToBackendMut::<BE>::to_backend_mut(dst);
            self.glwe_automorphism_assign(&mut dst_ref, &key, scratch);
        } else {
            let mut dst_ref = GLWEToBackendMut::<BE>::to_backend_mut(dst);
            let src_ref = GLWEToBackendRef::<BE>::to_backend_ref(src);
            self.glwe_automorphism(&mut dst_ref, &src_ref, &key, scratch);
        }

        dst.set_meta(src.meta());
        dst.set_log_budget(checked_log_budget_sub("rotate", dst.log_budget(), offset)?);
        Ok(())
    }

    fn ckks_rotate_assign_default<'s, Dst>(
        &self,
        dst: &mut Dst,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE>,
        Dst: CKKSCiphertextToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        let mut dst_ref = GLWEToBackendMut::<BE>::to_backend_mut(dst);
        self.glwe_automorphism_assign(&mut dst_ref, &key, scratch);
        Ok(())
    }
}

impl<BE: Backend> CKKSRotateDefault<BE> for Module<BE> {}
