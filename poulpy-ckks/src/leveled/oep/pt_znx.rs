use anyhow::Result;
use poulpy_core::{
    ScratchArenaTakeCore,
    layouts::{GLWEPlaintext, GLWEPlaintextToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Data, Module, ScratchArena},
};

use crate::CKKSPlaintextVecZnxToBackendMut;

use crate::{CKKSInfos, SetCKKSInfos, oep::CKKSImpl};

pub(crate) trait CKKSPlaintextZnxOep<BE: Backend + CKKSImpl<BE>> {
    fn ckks_extract_pt_znx_tmp_bytes(&self) -> usize
    where
        Self: VecZnxLshTmpBytes + VecZnxRshTmpBytes;

    fn ckks_extract_pt_znx<Dst, Src: Data, S>(
        &self,
        dst: &mut Dst,
        src: &GLWEPlaintext<Src>,
        src_meta: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSPlaintextVecZnxToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        S: CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        Self: VecZnxLshBackend<BE> + VecZnxRshBackend<BE>,
        GLWEPlaintext<Src>: GLWEPlaintextToBackendRef<BE>;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSPlaintextZnxOep<BE> for Module<BE> {
    fn ckks_extract_pt_znx_tmp_bytes(&self) -> usize
    where
        Self: VecZnxLshTmpBytes + VecZnxRshTmpBytes,
    {
        BE::ckks_extract_pt_znx_tmp_bytes(self)
    }

    fn ckks_extract_pt_znx<Dst, Src: Data, S>(
        &self,
        dst: &mut Dst,
        src: &GLWEPlaintext<Src>,
        src_meta: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSPlaintextVecZnxToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        S: CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        Self: VecZnxLshBackend<BE> + VecZnxRshBackend<BE>,
        GLWEPlaintext<Src>: GLWEPlaintextToBackendRef<BE>,
    {
        BE::ckks_extract_pt_znx(self, dst, src, src_meta, scratch)
    }
}
