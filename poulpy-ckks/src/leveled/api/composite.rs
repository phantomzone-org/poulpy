use anyhow::Result;
use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::{CKKSInfos, layouts::CKKSCiphertext, oep::CKKSImpl};

pub trait CKKSAddManyOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_add_many_tmp_bytes(&self) -> usize;

    fn ckks_add_many<Dst: Data, Src: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        inputs: &[&CKKSCiphertext<Src>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE>;
}

pub trait CKKSMulManyOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_many_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        T: GGLWEInfos;

    fn ckks_mul_many<Dst: Data, Src: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        inputs: &[&CKKSCiphertext<Src>],
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: GLWETensorKeyPreparedToBackendRef<BE>;
}

pub trait CKKSMulAddOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_add_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        T: GGLWEInfos;

    fn ckks_mul_add_pt_vec_znx_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos;

    fn ckks_mul_add_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos;

    fn ckks_mul_add_ct_into<Dst: Data, A: Data, B: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        CKKSCiphertext<B>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: GLWETensorKeyPreparedToBackendRef<BE>;

    fn ckks_mul_add_pt_vec_znx_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos;

    fn ckks_mul_add_pt_const_znx_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos;
}

pub trait CKKSMulSubOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_sub_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        T: GGLWEInfos;

    fn ckks_mul_sub_pt_vec_znx_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos;

    fn ckks_mul_sub_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos;

    fn ckks_mul_sub_ct_into<Dst: Data, A: Data, B: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        CKKSCiphertext<B>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: GLWETensorKeyPreparedToBackendRef<BE>;

    fn ckks_mul_sub_pt_vec_znx_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos;

    fn ckks_mul_sub_pt_const_znx_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos;
}

pub trait CKKSDotProductOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_dot_product_ct_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        T: GGLWEInfos;

    fn ckks_dot_product_pt_vec_znx_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos;

    fn ckks_dot_product_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos;

    fn ckks_dot_product_ct<Dst: Data, D: Data, E: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSCiphertext<E>],
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        CKKSCiphertext<E>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: GLWETensorKeyPreparedToBackendRef<BE>;

    fn ckks_dot_product_pt_vec_znx<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&E],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        E: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos;

    fn ckks_dot_product_pt_const_znx<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&E],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        E: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos;
}
