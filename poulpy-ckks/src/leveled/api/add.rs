use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEInfos, GLWEPlaintextToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddConstAssignBackend, VecZnxRshAddIntoBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Data, HostBackend, ScratchArena},
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

pub trait CKKSAddOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_add_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    fn ckks_add_into<Dst: Data, A: Data, B: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSCiphertext<B>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_assign<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_vec_znx_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_add_pt_vec_znx_into<Dst: Data, A: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddIntoBackend<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_vec_znx_assign<Dst: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddIntoBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_add_pt_vec_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        Self: ModuleN + VecZnxRshAddIntoBackend<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_add_pt_vec_rnx_assign<Dst: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        Self: ModuleN + VecZnxRshAddIntoBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_add_pt_const_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    fn ckks_add_pt_const_znx_into<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_const_znx_assign<Dst: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_const_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_add_pt_const_rnx_assign<Dst: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}

// Unnormalized variants of CKKS addition.
//
// # Safety
// Not a memory-safety hazard; `unsafe` flags a correctness invariant. Every
// method may leave the destination with non-K-normalized limbs — a
// normalizing op must eventually run before limbs overflow `i64` or the
// ciphertext is consumed by code requiring K-normalized inputs. Misuse
// yields silently-wrong plaintexts on decryption, never UB.
#[allow(clippy::missing_safety_doc)]
pub unsafe trait CKKSAddOpsUnsafe<BE: Backend> {
    unsafe fn ckks_add_into_unsafe<Dst: Data, A: Data, B: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSCiphertext<B>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    unsafe fn ckks_add_assign_unsafe<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    unsafe fn ckks_add_pt_vec_znx_into_unsafe<Dst: Data, A: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddIntoBackend<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    unsafe fn ckks_add_pt_vec_znx_assign_unsafe<Dst: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddIntoBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    unsafe fn ckks_add_pt_vec_rnx_into_unsafe<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        Self: ModuleN + VecZnxRshAddIntoBackend<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    unsafe fn ckks_add_pt_vec_rnx_assign_unsafe<Dst: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        Self: ModuleN + VecZnxRshAddIntoBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    unsafe fn ckks_add_pt_const_znx_into_unsafe<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    unsafe fn ckks_add_pt_const_znx_assign_unsafe<Dst: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    unsafe fn ckks_add_pt_const_rnx_into_unsafe<Dst: Data, A: Data, F>(
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

    unsafe fn ckks_add_pt_const_rnx_assign_unsafe<Dst: Data, F>(
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
