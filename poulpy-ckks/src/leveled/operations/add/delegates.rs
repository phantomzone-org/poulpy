use anyhow::Result;
use poulpy_core::{GLWEAdd, GLWEShift, ScratchTakeCore, layouts::GLWEInfos};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxRshAddInto, VecZnxRshTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    CKKSMeta,
    layouts::{
        CKKSCiphertext,
        plaintext::{
            CKKSConstPlaintextConversion, CKKSPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx, CKKSPlaintextVecRnx,
            CKKSPlaintextVecZnx,
        },
    },
    oep::CKKSImpl,
};

use super::{api::CKKSAddOps, oep::CKKSAddOep};

impl<BE: Backend + CKKSImpl<BE>> CKKSAddOps<BE> for Module<BE> {
    fn ckks_add_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        CKKSAddOep::ckks_add_tmp_bytes(self)
    }

    fn ckks_add(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSAddOep::ckks_add(self, dst, a, b, scratch)
    }

    fn ckks_add_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSAddOep::ckks_add_inplace(self, dst, a, scratch)
    }

    fn ckks_add_pt_vec_znx_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        CKKSAddOep::ckks_add_pt_vec_znx_tmp_bytes(self)
    }

    fn ckks_add_pt_vec_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddInto<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSAddOep::ckks_add_pt_vec_znx(self, dst, a, pt_znx, scratch)
    }

    fn ckks_add_pt_vec_znx_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddInto<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSAddOep::ckks_add_pt_vec_znx_inplace(self, dst, pt_znx, scratch)
    }

    fn ckks_add_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        CKKSAddOep::ckks_add_pt_vec_rnx_tmp_bytes(self, res, a, b)
    }

    fn ckks_add_pt_vec_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshAddInto<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        CKKSAddOep::ckks_add_pt_vec_rnx(self, dst, a, pt_rnx, prec, scratch)
    }

    fn ckks_add_pt_vec_rnx_inplace<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshAddInto<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        CKKSAddOep::ckks_add_pt_vec_rnx_inplace(self, dst, pt_rnx, prec, scratch)
    }

    fn ckks_add_const_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        CKKSAddOep::ckks_add_const_tmp_bytes(self)
    }

    fn ckks_add_pt_const_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSAddOep::ckks_add_pt_const_znx(self, dst, a, cst_znx, scratch)
    }

    fn ckks_add_pt_const_znx_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSAddOep::ckks_add_pt_const_znx_inplace(self, dst, cst_znx, scratch)
    }

    fn ckks_add_pt_const_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        CKKSAddOep::ckks_add_pt_const_rnx(self, dst, a, cst_rnx, prec, scratch)
    }

    fn ckks_add_pt_const_rnx_inplace<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        CKKSAddOep::ckks_add_pt_const_rnx_inplace(self, dst, cst_rnx, prec, scratch)
    }
}
