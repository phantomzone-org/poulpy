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

use crate::leveled::{api::CKKSPlaintextVecOps, oep::CKKSPlaintextZnxOep};

impl<BE: Backend + CKKSImpl<BE>> CKKSPlaintextVecOps<BE> for Module<BE>
where
    Module<BE>: VecZnxLshBackend<BE> + VecZnxLshTmpBytes + VecZnxRshBackend<BE> + VecZnxRshTmpBytes,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_extract_pt_tmp_bytes(&self) -> usize {
        CKKSPlaintextZnxOep::ckks_extract_pt_znx_tmp_bytes(self)
    }

    fn ckks_extract_pt<Dst, Src: Data, S>(
        &self,
        dst: &mut Dst,
        src: &GLWEPlaintext<Src>,
        src_meta: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSPlaintextVecZnxToBackendMut<BE> + CKKSInfos + SetCKKSInfos + LWEInfos,
        S: CKKSInfos,
        GLWEPlaintext<Src>: GLWEPlaintextToBackendRef<BE>,
    {
        CKKSPlaintextZnxOep::ckks_extract_pt_znx(self, dst, src, src_meta, scratch)
    }
}
