use anyhow::Result;
use poulpy_core::{
    ScratchArenaTakeCore,
    layouts::{GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::GLWEToBackendMut;

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

    fn ckks_extract_pt<D, S>(&self, dst: &mut D, src: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        D: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + LWEInfos,
        S: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        CKKSPlaintextZnxOep::ckks_extract_pt_znx(self, dst, src, scratch)
    }
}
