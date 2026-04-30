use anyhow::Result;
use poulpy_core::{
    ScratchArenaTakeCore,
    layouts::{
        GLWEPlaintext, GLWEPlaintextToBackendMut, GLWEPlaintextToBackendRef, GLWEToBackendMut, LWEInfos, glwe_backend_data_mut,
    },
};
use poulpy_hal::{
    api::{
        VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshAddIntoBackend, VecZnxRshBackend, VecZnxRshSubBackend, VecZnxRshTmpBytes,
    },
    layouts::{Backend, Data, Module, ScratchArena},
};

use crate::{
    CKKSInfos, ensure_base2k_match, ensure_plaintext_alignment,
    layouts::{CKKSCiphertext, CKKSPlaintextVecZnx},
};

pub(crate) trait CKKSPlaintextZnxDefault<BE: Backend> {
    fn ckks_add_pt_vec_znx_into_default<Dct, Dpt>(
        &self,
        ct: &mut CKKSCiphertext<Dct>,
        pt: &CKKSPlaintextVecZnx<Dpt>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dct: Data,
        Dpt: Data,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        Self: VecZnxRshAddIntoBackend<BE>,
        CKKSCiphertext<Dct>: GLWEToBackendMut<BE>,
        CKKSPlaintextVecZnx<Dpt>: GLWEPlaintextToBackendRef<BE>,
    {
        ensure_base2k_match("ckks_add_pt_vec_znx_into", ct.base2k().as_usize(), pt.base2k().as_usize())?;
        let offset = ensure_plaintext_alignment(
            "ckks_add_pt_vec_znx_into",
            ct.log_budget(),
            pt.log_delta(),
            pt.max_k().as_usize(),
        )?;
        let base2k = ct.base2k().as_usize();
        let mut ct_ref = GLWEToBackendMut::to_backend_mut(ct);
        let pt_ref = GLWEPlaintextToBackendRef::to_backend_ref(pt);
        let mut ct_data = glwe_backend_data_mut::<BE>(&mut ct_ref);
        self.vec_znx_rsh_add_into_backend(base2k, offset, &mut ct_data, 0, &pt_ref.data, 0, scratch);
        Ok(())
    }

    fn ckks_sub_pt_vec_znx_into_default<Dct, Dpt>(
        &self,
        ct: &mut CKKSCiphertext<Dct>,
        pt_znx: &CKKSPlaintextVecZnx<Dpt>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dct: Data,
        Dpt: Data,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        Self: VecZnxRshSubBackend<BE>,
        CKKSCiphertext<Dct>: GLWEToBackendMut<BE>,
        CKKSPlaintextVecZnx<Dpt>: GLWEPlaintextToBackendRef<BE>,
    {
        ensure_base2k_match("ckks_sub_pt_vec_znx_into", ct.base2k().as_usize(), pt_znx.base2k().as_usize())?;
        let offset = ensure_plaintext_alignment(
            "ckks_sub_pt_vec_znx_into",
            ct.log_budget(),
            pt_znx.log_delta(),
            pt_znx.max_k().as_usize(),
        )?;
        let base2k = ct.base2k().as_usize();
        let mut ct_ref = GLWEToBackendMut::to_backend_mut(ct);
        let pt_ref = GLWEPlaintextToBackendRef::to_backend_ref(pt_znx);
        let mut ct_data = glwe_backend_data_mut::<BE>(&mut ct_ref);
        self.vec_znx_rsh_sub_backend(base2k, offset, &mut ct_data, 0, &pt_ref.data, 0, scratch);
        Ok(())
    }

    fn ckks_extract_pt_znx_tmp_bytes_default(&self) -> usize
    where
        Self: VecZnxLshTmpBytes + VecZnxRshTmpBytes,
    {
        self.vec_znx_rsh_tmp_bytes().max(self.vec_znx_lsh_tmp_bytes())
    }

    fn ckks_extract_pt_znx_default<Dst, Src, S>(
        &self,
        dst: &mut CKKSPlaintextVecZnx<Dst>,
        src: &GLWEPlaintext<Src>,
        src_meta: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Src: Data,
        S: CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        Self: VecZnxLshBackend<BE> + VecZnxRshBackend<BE>,
        CKKSPlaintextVecZnx<Dst>: GLWEPlaintextToBackendMut<BE>,
        GLWEPlaintext<Src>: GLWEPlaintextToBackendRef<BE>,
    {
        ensure_base2k_match("ckks_extract_pt_znx", src.base2k().as_usize(), dst.base2k().as_usize())?;
        let available = src_meta.log_budget() + dst.log_delta();
        if available < dst.effective_k() {
            return Err(crate::CKKSCompositionError::PlaintextAlignmentImpossible {
                op: "ckks_extract_pt_znx",
                ct_log_budget: src_meta.log_budget(),
                pt_log_delta: dst.log_delta(),
                pt_max_k: dst.max_k().as_usize(),
            }
            .into());
        }
        let dst_k = dst.max_k().as_usize();
        let dst_base2k: usize = dst.base2k().into();
        let mut dst_ref = GLWEPlaintextToBackendMut::to_backend_mut(dst);
        let src_ref = GLWEPlaintextToBackendRef::to_backend_ref(src);
        if available < dst_k {
            self.vec_znx_rsh_backend(dst_base2k, dst_k - available, &mut dst_ref.data, 0, &src_ref.data, 0, scratch);
        } else if available > dst_k {
            self.vec_znx_lsh_backend(dst_base2k, available - dst_k, &mut dst_ref.data, 0, &src_ref.data, 0, scratch);
        } else {
            self.vec_znx_rsh_backend(dst_base2k, 0, &mut dst_ref.data, 0, &src_ref.data, 0, scratch);
        }
        Ok(())
    }
}

impl<BE: Backend> CKKSPlaintextZnxDefault<BE> for Module<BE> {}
