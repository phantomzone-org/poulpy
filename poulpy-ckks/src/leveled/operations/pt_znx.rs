use poulpy_core::{
    ScratchTakeCore,
    layouts::{GLWEPlaintextToMut, GLWEPlaintextToRef, GLWEToMut, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxRshAddInto, VecZnxRshSub},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    CKKSInfos, ensure_base2k_match, ensure_plaintext_alignment,
    layouts::{CKKSCiphertext, CKKSPlaintextVecZnx},
};
use anyhow::Result;

pub trait CKKSPlaintextZnxOps<BE: Backend> {
    fn ckks_add_pt_vec_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxRshAddInto<BE>;
    fn ckks_sub_pt_vec_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxRshSub<BE>;
    fn ckks_extract_pt_znx(
        &self,
        dst: &mut CKKSPlaintextVecZnx<impl DataMut>,
        src: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxLsh<BE>;
}

impl<BE: Backend> CKKSPlaintextZnxOps<BE> for Module<BE> {
    fn ckks_add_pt_vec_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxRshAddInto<BE>,
    {
        ensure_base2k_match("ckks_add_pt_vec_znx", dst.base2k().as_usize(), src.base2k().as_usize())?;
        let offset = ensure_plaintext_alignment(
            "ckks_add_pt_vec_znx",
            dst.log_hom_rem(),
            src.log_decimal(),
            src.max_k().as_usize(),
        )?;
        let dst = &mut GLWEToMut::to_mut(dst);
        let src = &GLWEPlaintextToRef::to_ref(src);
        let base2k: usize = dst.base2k().into();
        self.vec_znx_rsh_add_into(base2k, offset, dst.data_mut(), 0, src.data(), 0, scratch);
        Ok(())
    }

    fn ckks_extract_pt_znx(
        &self,
        dst: &mut CKKSPlaintextVecZnx<impl DataMut>,
        src: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxLsh<BE>,
    {
        let offset = ensure_plaintext_alignment(
            "ckks_extract_pt_znx",
            src.log_hom_rem(),
            dst.log_decimal(),
            dst.max_k().as_usize(),
        )?;
        let dst = &mut GLWEPlaintextToMut::to_mut(dst);
        let src = &GLWEPlaintextToRef::to_ref(src);
        let base2k: usize = dst.base2k().into();
        self.vec_znx_lsh(base2k, offset, dst.data_mut(), 0, src.data(), 0, scratch);
        Ok(())
    }

    fn ckks_sub_pt_vec_znx(
        &self,
        ct: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxRshSub<BE>,
    {
        ensure_base2k_match("ckks_sub_pt_vec_znx", ct.base2k().as_usize(), pt_znx.base2k().as_usize())?;
        let offset = ensure_plaintext_alignment(
            "ckks_sub_pt_vec_znx",
            ct.log_hom_rem(),
            pt_znx.log_decimal(),
            pt_znx.max_k().as_usize(),
        )?;
        let ct = &mut GLWEToMut::to_mut(ct);
        let pt_znx = &GLWEPlaintextToRef::to_ref(pt_znx);
        let base2k: usize = ct.base2k().into();
        self.vec_znx_rsh_sub(base2k, offset, ct.data_mut(), 0, pt_znx.data(), 0, scratch);
        Ok(())
    }
}
