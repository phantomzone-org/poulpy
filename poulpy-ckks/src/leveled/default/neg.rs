use anyhow::Result;
use poulpy_core::{
    GLWENegate, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSCiphertextToBackendMut, CKKSCiphertextToBackendRef};

use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub, ckks_offset_unary};

pub(crate) trait CKKSNegDefault<BE: Backend> {
    fn ckks_neg_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_neg_into_default<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWENegate<BE> + GLWEShift<BE>,
        Dst: CKKSCiphertextToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: CKKSCiphertextToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let offset = ckks_offset_unary(dst, src);
        if offset != 0 {
            self.glwe_lsh(dst, src, offset, scratch);
            dst.set_meta(src.meta());
            dst.set_log_budget(checked_log_budget_sub("neg", src.log_budget(), offset)?);
            {
                let mut dst_ref = GLWEToBackendMut::<BE>::to_backend_mut(dst);
                self.glwe_negate_assign(&mut dst_ref);
            }
        } else {
            {
                let mut dst_ref = GLWEToBackendMut::<BE>::to_backend_mut(dst);
                let src_ref = GLWEToBackendRef::<BE>::to_backend_ref(src);
                self.glwe_negate(&mut dst_ref, &src_ref);
            }
            dst.set_meta(src.meta());
        }
        Ok(())
    }

    fn ckks_neg_assign_default<Dst>(&self, dst: &mut Dst) -> Result<()>
    where
        Self: GLWENegate<BE>,
        Dst: CKKSCiphertextToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        let mut dst_ref = GLWEToBackendMut::<BE>::to_backend_mut(dst);
        self.glwe_negate_assign(&mut dst_ref);
        Ok(())
    }
}

impl<BE: Backend> CKKSNegDefault<BE> for Module<BE> {}
