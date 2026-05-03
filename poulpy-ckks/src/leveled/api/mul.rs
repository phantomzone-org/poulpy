use anyhow::Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, prepared::GLWETensorKeyPreparedToBackendRef};
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSInfos, SetCKKSInfos, oep::CKKSImpl};

pub trait CKKSMulOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        T: GGLWEInfos;

    fn ckks_square_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        T: GGLWEInfos;

    fn ckks_mul_pt_vec_znx_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos;

    fn ckks_mul_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos;

    fn ckks_mul_into<Dst, A, B, T>(&self, dst: &mut Dst, a: &A, b: &B, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    fn ckks_mul_assign<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + LWEInfos + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    fn ckks_square_into<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    fn ckks_square_assign<Dst, T>(&self, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + LWEInfos + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    fn ckks_mul_pt_vec_znx_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos;

    fn ckks_mul_pt_vec_znx_assign<Dst, P>(&self, dst: &mut Dst, pt_znx: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + LWEInfos + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos;

    fn ckks_mul_pt_const_znx_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos;

    fn ckks_mul_pt_const_znx_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + LWEInfos + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos;
}
