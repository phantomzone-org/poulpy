use anyhow::Result;
use poulpy_core::{ScratchArenaTakeCore, layouts::GLWEPlaintext};
use poulpy_hal::{
    api::{VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Data, ScratchArena},
};

use crate::{CKKSInfos, layouts::CKKSPlaintextVecZnx, oep::CKKSImpl};

pub trait CKKSPlaintextZnxOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_extract_pt_znx_tmp_bytes(&self) -> usize
    where
        Self: VecZnxLshTmpBytes + VecZnxRshTmpBytes;

    fn ckks_extract_pt_znx<Dst: Data, Src: Data, S>(
        &self,
        dst: &mut CKKSPlaintextVecZnx<Dst>,
        src: &GLWEPlaintext<Src>,
        src_meta: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        Self: VecZnxLshBackend<BE> + VecZnxRshBackend<BE>,
        CKKSPlaintextVecZnx<Dst>: poulpy_core::layouts::GLWEPlaintextToBackendMut<BE>,
        GLWEPlaintext<Src>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>;
}
