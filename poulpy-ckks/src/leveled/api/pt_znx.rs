use anyhow::Result;
use poulpy_core::layouts::{GLWEPlaintext, GLWEPlaintextToBackendRef, LWEInfos};
use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::CKKSPlaintextVecZnxToBackendMut;

use crate::{CKKSInfos, SetCKKSInfos, oep::CKKSImpl};

pub trait CKKSPlaintextVecOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_extract_pt_tmp_bytes(&self) -> usize;

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
        GLWEPlaintext<Src>: GLWEPlaintextToBackendRef<BE>;
}
