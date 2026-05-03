use anyhow::Result;
use poulpy_core::{
    ScratchArenaTakeCore,
    layouts::{GLWEInfos, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::GLWEToBackendMut;

use crate::{CKKSInfos, SetCKKSInfos, oep::CKKSImpl};

pub(crate) trait CKKSPlaintextZnxOep<BE: Backend + CKKSImpl<BE>> {
    fn ckks_extract_pt_znx_tmp_bytes(&self) -> usize
    where
        Self: VecZnxLshTmpBytes + VecZnxRshTmpBytes;

    fn ckks_extract_pt_znx<D, S>(&self, dst: &mut D, src: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        D: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        S: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        Self: VecZnxLshBackend<BE> + VecZnxRshBackend<BE>;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSPlaintextZnxOep<BE> for Module<BE> {
    fn ckks_extract_pt_znx_tmp_bytes(&self) -> usize
    where
        Self: VecZnxLshTmpBytes + VecZnxRshTmpBytes,
    {
        BE::ckks_extract_pt_znx_tmp_bytes(self)
    }

    fn ckks_extract_pt_znx<D, S>(&self, dst: &mut D, src: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        D: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        S: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        Self: VecZnxLshBackend<BE> + VecZnxRshBackend<BE>,
    {
        BE::ckks_extract_pt_znx(self, dst, src, scratch)
    }
}
