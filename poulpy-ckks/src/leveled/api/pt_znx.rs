use anyhow::Result;
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendRef, LWEInfos};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::GLWEToBackendMut;

use crate::{CKKSInfos, SetCKKSInfos, oep::CKKSImpl};

pub trait CKKSPlaintextVecOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_extract_pt_tmp_bytes(&self) -> usize;

    fn ckks_extract_pt<D, S>(&self, dst: &mut D, src: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        D: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + LWEInfos,
        S: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos;
}
