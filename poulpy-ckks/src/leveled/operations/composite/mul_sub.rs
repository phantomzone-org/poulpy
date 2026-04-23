//! CKKS fused multiply-sub: `dst -= a · b`.

use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWEShift, GLWESub, GLWETensoring, ScratchTakeCore,
    layouts::{GGLWEInfos, GLWE, GLWEInfos, GLWETensorKeyPrepared},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxRshTmpBytes},
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
    leveled::operations::{mul::CKKSMulOps, sub::CKKSSubOps},
    oep::CKKSImpl,
};

pub trait CKKSMulSubOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_sub_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + VecZnxRshTmpBytes + CKKSSubOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_sub_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + VecZnxRshTmpBytes + CKKSSubOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_sub_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + VecZnxRshTmpBytes + CKKSSubOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_sub_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + VecZnxRshTmpBytes + CKKSSubOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_sub_ct(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE> + GLWETensoring<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_sub_pt_vec_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_sub_pt_vec_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWESub + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_mul_sub_const_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWESub + GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_sub_const_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWESub + GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}

fn take_mul_tmp<'a, BE: Backend, D: DataMut>(
    dst: &CKKSCiphertext<D>,
    scratch: &'a mut Scratch<BE>,
) -> (CKKSCiphertext<&'a mut [u8]>, &'a mut Scratch<BE>)
where
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let layout = dst.glwe_layout();
    let (tmp, scratch_r) = scratch.take_glwe(&layout);
    (CKKSCiphertext::from_inner(tmp, CKKSMeta::default()), scratch_r)
}

impl<BE: Backend + CKKSImpl<BE>> CKKSMulSubOps<BE> for Module<BE> {
    fn ckks_mul_sub_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + VecZnxRshTmpBytes + CKKSSubOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_tmp_bytes(res, tsk).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + VecZnxRshTmpBytes + CKKSSubOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_znx_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + VecZnxRshTmpBytes + CKKSSubOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_rnx_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + VecZnxRshTmpBytes + CKKSSubOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_const_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_ct(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE> + GLWETensoring<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul(&mut tmp, a, b, tsk, scratch_r)?;
        self.ckks_sub_inplace(dst, &tmp, scratch_r)
    }

    fn ckks_mul_sub_pt_vec_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul_pt_vec_znx(&mut tmp, a, pt_znx, scratch_r)?;
        self.ckks_sub_inplace(dst, &tmp, scratch_r)
    }

    fn ckks_mul_sub_pt_vec_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWESub + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul_pt_vec_rnx(&mut tmp, a, pt_rnx, prec, scratch_r)?;
        self.ckks_sub_inplace(dst, &tmp, scratch_r)
    }

    fn ckks_mul_sub_const_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWESub + GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul_pt_const_znx(&mut tmp, a, cst_znx, scratch_r)?;
        self.ckks_sub_inplace(dst, &tmp, scratch_r)
    }

    fn ckks_mul_sub_const_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWESub + GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul_pt_const_rnx(&mut tmp, a, cst_rnx, prec, scratch_r)?;
        self.ckks_sub_inplace(dst, &tmp, scratch_r)
    }
}
