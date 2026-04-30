use anyhow::Result;
use poulpy_core::{
    GLWEShift, GLWESub, ScratchArenaTakeCore,
    layouts::{GLWEInfos, GLWEPlaintextToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxRshSubBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Data, HostBackend, Module, ScratchArena},
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

pub(crate) trait CKKSSubOep<BE: Backend + CKKSImpl<BE>> {
    fn ckks_sub_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_pt_vec_znx_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_into<Dst: Data, A: Data, B: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWESub<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSCiphertext<B>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_assign<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWESub<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_vec_znx_into<Dst: Data, A: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSubBackend<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_vec_znx_assign<Dst: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSubBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_pt_vec_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        Self: ModuleN + VecZnxRshSubBackend<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_sub_pt_vec_rnx_assign<Dst: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        Self: ModuleN + VecZnxRshSubBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_sub_pt_const_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    fn ckks_sub_pt_const_znx_into<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_const_znx_assign<Dst: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_const_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_sub_pt_const_rnx_assign<Dst: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSSubOep<BE> for Module<BE> {
    fn ckks_sub_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        BE::ckks_sub_tmp_bytes(self)
    }

    fn ckks_sub_pt_vec_znx_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        BE::ckks_sub_pt_vec_znx_tmp_bytes(self)
    }

    fn ckks_sub_into<Dst: Data, A: Data, B: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWESub<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSCiphertext<B>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_sub_into(self, dst, a, b, scratch)
    }

    fn ckks_sub_assign<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWESub<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_sub_assign(self, dst, a, scratch)
    }

    fn ckks_sub_pt_vec_znx_into<Dst: Data, A: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSubBackend<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_sub_pt_vec_znx_into(self, dst, a, pt_znx, scratch)
    }

    fn ckks_sub_pt_vec_znx_assign<Dst: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSubBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_sub_pt_vec_znx_assign(self, dst, pt_znx, scratch)
    }

    fn ckks_sub_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        BE::ckks_sub_pt_vec_rnx_tmp_bytes(self, res, a, b)
    }

    fn ckks_sub_pt_vec_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        Self: ModuleN + VecZnxRshSubBackend<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        BE::ckks_sub_pt_vec_rnx_into(self, dst, a, pt_rnx, prec, scratch)
    }

    fn ckks_sub_pt_vec_rnx_assign<Dst: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        Self: ModuleN + VecZnxRshSubBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        BE::ckks_sub_pt_vec_rnx_assign(self, dst, pt_rnx, prec, scratch)
    }

    fn ckks_sub_pt_const_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        BE::ckks_sub_pt_const_tmp_bytes(self)
    }

    fn ckks_sub_pt_const_znx_into<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_sub_pt_const_znx_into(self, dst, a, cst_znx, scratch)
    }

    fn ckks_sub_pt_const_znx_assign<Dst: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_sub_pt_const_znx_assign(self, dst, cst_znx, scratch)
    }

    fn ckks_sub_pt_const_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        BE::ckks_sub_pt_const_rnx_into(self, dst, a, cst_rnx, prec, scratch)
    }

    fn ckks_sub_pt_const_rnx_assign<Dst: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        BE::ckks_sub_pt_const_rnx_assign(self, dst, cst_rnx, prec, scratch)
    }
}
