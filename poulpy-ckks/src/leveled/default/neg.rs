use anyhow::Result;
use poulpy_core::{
    GLWENegate, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, Module, ScratchArena},
};

use crate::{
    CKKSInfos, checked_log_budget_sub,
    layouts::{CKKSCiphertext, ciphertext::CKKSOffset},
};

pub(crate) trait CKKSNegDefault<BE: Backend> {
    fn ckks_neg_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_neg_into_default<Dst, Src>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Src: Data,
        Self: GLWENegate<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let offset = dst.offset_unary(src);
        if offset != 0 {
            self.glwe_lsh(dst, src, offset, scratch);
            dst.meta = src.meta();
            dst.meta.log_budget = checked_log_budget_sub("neg", src.log_budget(), offset)?;
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
            dst.meta = src.meta();
        }
        Ok(())
    }

    fn ckks_neg_assign_default<Dst>(&self, dst: &mut CKKSCiphertext<Dst>) -> Result<()>
    where
        Dst: Data,
        Self: GLWENegate<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
    {
        let mut dst_ref = GLWEToBackendMut::<BE>::to_backend_mut(dst);
        self.glwe_negate_assign(&mut dst_ref);
        Ok(())
    }
}

impl<BE: Backend> CKKSNegDefault<BE> for Module<BE> {}
