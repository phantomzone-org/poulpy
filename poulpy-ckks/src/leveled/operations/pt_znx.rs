use poulpy_core::{
    ScratchTakeCore,
    layouts::{GLWEToMut, GLWEToRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxRshAddInto, VecZnxRshSub},
    layouts::{Backend, Module, Scratch},
};

use crate::CKKSInfos;
use anyhow::Result;

pub trait CKKSPlaintextZnxOps<BE: Backend> {
    fn ckks_add_pt_znx<D, S>(&self, dst: &mut D, src: &S, scratch: &mut Scratch<BE>) -> Result<()>
    where
        D: GLWEToMut + LWEInfos + CKKSInfos,
        S: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxRshAddInto<BE>;
    fn ckks_sub_pt_znx<D, S>(&self, dst: &mut D, src: &S, scratch: &mut Scratch<BE>) -> Result<()>
    where
        D: GLWEToMut + LWEInfos + CKKSInfos,
        S: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxRshSub<BE>;
    fn ckks_extract_pt_znx<D, S>(&self, dst: &mut D, src: &S, scratch: &mut Scratch<BE>) -> Result<()>
    where
        D: GLWEToMut + LWEInfos + CKKSInfos,
        S: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxLsh<BE>;
}

impl<BE: Backend> CKKSPlaintextZnxOps<BE> for Module<BE> {
    fn ckks_add_pt_znx<D, S>(&self, dst: &mut D, src: &S, scratch: &mut Scratch<BE>) -> Result<()>
    where
        D: GLWEToMut + LWEInfos + CKKSInfos,
        S: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxRshAddInto<BE>,
    {
        anyhow::ensure!(dst.base2k() == src.base2k());
        anyhow::ensure!(
            dst.log_hom_rem() + src.log_decimal() >= src.max_k().as_usize(),
            "max_k must be <= log_hom_rem + log_decimal (max_k={}, log_hom_rem={}, log_decimal={})",
            src.max_k().as_usize(),
            dst.log_hom_rem(),
            src.log_decimal()
        );
        let offset = dst.log_hom_rem() + src.log_decimal() - src.max_k().as_usize();
        let dst = &mut dst.to_mut();
        let src = &src.to_ref();
        let base2k: usize = dst.base2k().into();
        self.vec_znx_rsh_add_into(base2k, offset, dst.data_mut(), 0, src.data(), 0, scratch);
        Ok(())
    }

    fn ckks_extract_pt_znx<D, S>(&self, dst: &mut D, src: &S, scratch: &mut Scratch<BE>) -> Result<()>
    where
        D: GLWEToMut + LWEInfos + CKKSInfos,
        S: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxLsh<BE>,
    {
        anyhow::ensure!(
            src.log_hom_rem() + dst.log_decimal() >= dst.max_k().as_usize(),
            "max_k must be <= log_hom_rem + log_decimal (max_k={}, log_hom_rem={}, log_decimal={})",
            dst.max_k().as_usize(),
            src.log_hom_rem(),
            dst.log_decimal()
        );
        let offset = src.log_hom_rem() + dst.log_decimal() - dst.max_k().as_usize();
        let dst = &mut dst.to_mut();
        let src = &src.to_ref();
        let base2k: usize = dst.base2k().into();
        self.vec_znx_lsh(base2k, offset, dst.data_mut(), 0, src.data(), 0, scratch);
        Ok(())
    }

    fn ckks_sub_pt_znx<C, P>(&self, ct: &mut C, pt_znx: &P, scratch: &mut Scratch<BE>) -> Result<()>
    where
        C: GLWEToMut + LWEInfos + CKKSInfos,
        P: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxRshSub<BE>,
    {
        anyhow::ensure!(ct.base2k() == pt_znx.base2k());
        anyhow::ensure!(
            ct.log_hom_rem() + pt_znx.log_decimal() >= pt_znx.max_k().as_usize(),
            "max_k must be <= log_hom_rem + log_decimal (max_k={}, log_hom_rem={}, log_decimal={})",
            pt_znx.max_k().as_usize(),
            ct.log_hom_rem(),
            pt_znx.log_decimal()
        );
        let offset = ct.log_hom_rem() + pt_znx.log_decimal() - pt_znx.max_k().as_usize();
        let ct = &mut ct.to_mut();
        let pt_znx = &pt_znx.to_ref();
        let base2k: usize = ct.base2k().into();
        self.vec_znx_rsh_sub(base2k, offset, ct.data_mut(), 0, pt_znx.data(), 0, scratch);
        Ok(())
    }
}
