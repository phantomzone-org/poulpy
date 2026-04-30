use std::collections::HashMap;

use poulpy_hal::{
    api::{
        ModuleLogN, ModuleN, ScratchAvailable, VecZnxAddAssignBackend, VecZnxAddIntoBackend, VecZnxCopyBackend,
        VecZnxLshAddIntoBackend, VecZnxLshBackend, VecZnxLshAssignBackend, VecZnxLshSubBackend, VecZnxLshTmpBytes,
        VecZnxMulXpMinusOneBackend, VecZnxMulXpMinusOneAssignBackend, VecZnxNegateBackend, VecZnxNegateAssignBackend,
        VecZnxNormalize, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes, VecZnxRotateBackend, VecZnxRotateAssignBackend,
        VecZnxRotateAssignTmpBytes, VecZnxRshAssignBackend, VecZnxRshTmpBytes, VecZnxSubBackend, VecZnxSubAssignBackend,
        VecZnxSubNegateAssignBackend, VecZnxZeroBackend,
    },
    layouts::{Backend, Data, GaloisElement, ScratchArena},
};

use crate::{
    ScratchArenaTakeCore,
    api::GLWEAutomorphism,
    glwe_packer::{GLWEPacker, pack_core},
    layouts::{
        GGLWEInfos, GGSWBackendMut, GGSWBackendRef, GGSWInfos, GLWE, GLWEAutomorphismKeyHelper, GLWEBackendMut, GLWEBackendRef,
        GLWEInfos, GLWEPlaintext, GLWETensor, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
        LWEInfos, ModuleCoreAlloc, ggsw_at_backend_mut_from_mut, ggsw_at_backend_ref_from_ref, glwe_backend_mut_from_mut,
        glwe_backend_ref_from_mut, glwe_backend_ref_from_ref, prepared::GGLWEPreparedToBackendRef,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};

pub trait GLWETrace<BE: Backend> {
    fn glwe_trace_galois_elements(&self) -> Vec<i64>;

    fn glwe_trace_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_trace<'s, R, A, K, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;

    fn glwe_trace_assign<'s, R, K, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;
}

pub trait GLWEPacking<BE: Backend> {
    fn glwe_pack_galois_elements(&self) -> Vec<i64>;

    fn glwe_pack_tmp_bytes<R, K>(&self, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_pack<'s, R, A, K, H>(
        &self,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: crate::layouts::GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;
}

pub trait GLWEPackerOps<BE: Backend>
where
    Self: Sized
        + ModuleLogN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + GLWEAutomorphism<BE>
        + GaloisElement
        + GLWERotate<BE>
        + GLWESub<BE>
        + GLWEShift<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + GLWECopy<BE>,
{
    fn packer_add<'s, A, K, H>(
        &self,
        packer: &mut GLWEPacker<BE::OwnedBuf>,
        a: Option<&A>,
        i: usize,
        auto_keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        A: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        crate::layouts::BackendGLWE<BE>: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEToBackendRef<BE>,
        BE: 's,
    {
        pack_core(self, a, &mut packer.accumulators, i, auto_keys, scratch)
    }
}

pub trait GLWEMulConst<BE: Backend> {
    fn glwe_mul_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b_size: usize) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    fn glwe_mul_const<'s, R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        b: &[i64],
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWE<A>: GLWEToBackendRef<BE>;

    fn glwe_mul_const_assign<'s, R>(&self, cnv_offset: usize, res: &mut GLWE<R>, b: &[i64], scratch: &mut ScratchArena<'s, BE>)
    where
        R: Data,
        GLWE<R>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>;
}

pub trait GLWEMulPlain<BE: Backend> {
    fn glwe_mul_plain_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain<'s, R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWEPlaintext<B>,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWE<A>: GLWEToBackendRef<BE>,
        GLWEPlaintext<B>: crate::layouts::GLWEPlaintextToBackendRef<BE>;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain_assign<'s, R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        res_effective_k: usize,
        a: &GLWEPlaintext<A>,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        GLWE<R>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        GLWEPlaintext<A>: crate::layouts::GLWEPlaintextToBackendRef<BE>;
}

pub trait GLWETensoring<BE: Backend> {
    fn glwe_tensor_apply_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_tensor_square_apply_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        self.glwe_tensor_apply_tmp_bytes(res, a, a)
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply<'s, R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWETensor<R>: GLWEToBackendMut<BE>,
        GLWE<A>: crate::layouts::GLWEToBackendRef<BE>,
        GLWE<B>: crate::layouts::GLWEToBackendRef<BE>;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_square_apply<'s, R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        GLWETensor<R>: GLWEToBackendMut<BE>,
        GLWE<A>: crate::layouts::GLWEToBackendRef<BE>;

    fn glwe_tensor_relinearize<'s, R, A, B>(
        &self,
        res: &mut GLWE<R>,
        a: &GLWETensor<A>,
        tsk: &GLWETensorKeyPrepared<B, BE>,
        tsk_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWETensorKeyPrepared<B, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
        GLWETensor<A>: crate::layouts::GLWEToBackendRef<BE>;

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(&self, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;
}

pub trait GLWEAdd<BE: Backend>
where
    Self: ModuleN + VecZnxAddIntoBackend<BE> + VecZnxCopyBackend<BE> + VecZnxAddAssignBackend<BE> + VecZnxZeroBackend<BE>,
{
    fn glwe_add_into<R, A, B>(&self, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        B: GLWEToBackendRef<BE>,
    {
        let res = &mut res.to_backend_mut();
        let a = &a.to_backend_ref();
        let b = &b.to_backend_ref();

        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(b.n(), self.n() as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.base2k(), b.base2k());
        assert_eq!(res.base2k(), b.base2k());

        if a.rank() == 0 {
            assert_eq!(res.rank(), b.rank());
        } else if b.rank() == 0 {
            assert_eq!(res.rank(), a.rank());
        } else {
            assert_eq!(res.rank(), a.rank());
            assert_eq!(res.rank(), b.rank());
        }

        let min_col: usize = (a.rank().min(b.rank()) + 1).into();
        let max_col: usize = (a.rank().max(b.rank()) + 1).into();
        let self_col: usize = (res.rank() + 1).into();

        for i in 0..min_col {
            self.vec_znx_add_into_backend(&mut res.data, i, &a.data, i, &b.data, i);
        }

        if a.rank() > b.rank() {
            for i in min_col..max_col {
                self.vec_znx_copy_backend(&mut res.data, i, &a.data, i);
            }
        } else {
            for i in min_col..max_col {
                self.vec_znx_copy_backend(&mut res.data, i, &b.data, i);
            }
        }

        for i in max_col..self_col {
            self.vec_znx_zero_backend(&mut res.data, i);
        }
    }

    fn glwe_add_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();
        self.glwe_add_assign_backend(&mut res, &a);
    }

    fn glwe_add_assign_backend(&self, res: &mut GLWEBackendMut<'_, BE>, a: &GLWEBackendRef<'_, BE>) {
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        for i in 0..(a.rank() + 1).into() {
            self.vec_znx_add_assign_backend(&mut res.data, i, &a.data, i);
        }
    }
}

pub trait GLWENegate<BE: Backend>
where
    Self: VecZnxNegateBackend<BE> + VecZnxNegateAssignBackend<BE> + ModuleN,
{
    fn glwe_negate(&self, res: &mut GLWEBackendMut<'_, BE>, a: &GLWEBackendRef<'_, BE>) {
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.rank(), res.rank());
        let cols = res.rank().as_usize() + 1;
        for i in 0..cols {
            self.vec_znx_negate_backend(&mut res.data, i, &a.data, i);
        }
        res.base2k = a.base2k;
    }

    fn glwe_negate_assign(&self, res: &mut GLWEBackendMut<'_, BE>) {
        assert_eq!(res.n(), self.n() as u32);
        let cols = res.rank().as_usize() + 1;
        for i in 0..cols {
            self.vec_znx_negate_assign_backend(&mut res.data, i);
        }
    }
}

pub trait GLWESub<BE: Backend>
where
    Self: ModuleN
        + VecZnxSubBackend<BE>
        + VecZnxSubAssignBackend<BE>
        + VecZnxSubNegateAssignBackend<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxNegateBackend<BE>
        + VecZnxZeroBackend<BE>,
{
    fn glwe_sub_backend<'a>(&self, res: &mut GLWEBackendMut<'_, BE>, a: &GLWEBackendRef<'a, BE>, b: &GLWEBackendRef<'a, BE>) {
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(b.n(), self.n() as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.base2k(), res.base2k());
        assert_eq!(b.base2k(), res.base2k());

        if a.rank() == 0 {
            assert_eq!(res.rank(), b.rank());
        } else if b.rank() == 0 {
            assert_eq!(res.rank(), a.rank());
        } else {
            assert_eq!(res.rank(), a.rank());
            assert_eq!(res.rank(), b.rank());
        }

        let min_col: usize = (a.rank().min(b.rank()) + 1).into();
        let max_col: usize = (a.rank().max(b.rank()) + 1).into();
        let self_col: usize = (res.rank() + 1).into();

        for i in 0..min_col {
            self.vec_znx_sub_backend(&mut res.data, i, &a.data, i, &b.data, i);
        }

        if a.rank() > b.rank() {
            for i in min_col..max_col {
                self.vec_znx_copy_backend(&mut res.data, i, &a.data, i);
            }
        } else {
            for i in min_col..max_col {
                self.vec_znx_negate_backend(&mut res.data, i, &b.data, i);
            }
        }

        for i in max_col..self_col {
            self.vec_znx_zero_backend(&mut res.data, i);
        }
    }

    fn glwe_sub<R, A, B>(&self, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        B: GLWEToBackendRef<BE>,
    {
        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();
        let b = b.to_backend_ref();
        self.glwe_sub_backend(&mut res, &a, &b);
    }

    fn glwe_sub_assign_backend(&self, res: &mut GLWEBackendMut<'_, BE>, a: &GLWEBackendRef<'_, BE>) {
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() == a.rank() || a.rank() == 0);

        for i in 0..(a.rank() + 1).into() {
            self.vec_znx_sub_assign_backend(&mut res.data, i, &a.data, i);
        }
    }

    fn glwe_sub_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        self.glwe_sub_assign_backend(&mut res.to_backend_mut(), &a.to_backend_ref());
    }

    fn glwe_sub_negate_assign_backend(&self, res: &mut GLWEBackendMut<'_, BE>, a: &GLWEBackendRef<'_, BE>) {
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() == a.rank() || a.rank() == 0);

        for i in 0..(a.rank() + 1).into() {
            self.vec_znx_sub_negate_assign_backend(&mut res.data, i, &a.data, i);
        }
    }

    fn glwe_sub_negate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        self.glwe_sub_negate_assign_backend(&mut res.to_backend_mut(), &a.to_backend_ref());
    }
}

pub trait GLWERotate<BE: Backend>
where
    Self:
        ModuleN + VecZnxRotateBackend<BE> + VecZnxRotateAssignBackend<BE> + VecZnxRotateAssignTmpBytes + VecZnxZeroBackend<BE>,
{
    fn glwe_rotate_tmp_bytes(&self) -> usize {
        self.vec_znx_rotate_assign_tmp_bytes()
    }

    fn glwe_rotate<'r, 'a>(&self, k: i64, res: &mut GLWEBackendMut<'r, BE>, a: &GLWEBackendRef<'a, BE>) {
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert!(res.rank() == a.rank() || a.rank() == 0);

        let res_cols = (res.rank() + 1).into();
        let a_cols = (a.rank() + 1).into();

        for i in 0..a_cols {
            self.vec_znx_rotate_backend(k, &mut res.data, i, &a.data, i);
        }
        for i in a_cols..res_cols {
            self.vec_znx_zero_backend(&mut res.data, i);
        }
    }

    fn glwe_rotate_assign<'s, 'r>(&self, k: i64, res: &mut GLWEBackendMut<'r, BE>, scratch: &mut ScratchArena<'s, BE>)
    where
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        assert!(
            scratch.available() >= self.glwe_rotate_tmp_bytes(),
            "scratch.available(): {} < GLWERotate::glwe_rotate_tmp_bytes: {}",
            scratch.available(),
            self.glwe_rotate_tmp_bytes()
        );

        for i in 0..(res.rank() + 1).into() {
            self.vec_znx_rotate_assign_backend(k, &mut res.data, i, &mut scratch.borrow());
        }
    }
}

pub trait GGSWRotate<BE: Backend>
where
    Self: GLWERotate<BE>,
{
    fn ggsw_rotate_tmp_bytes(&self) -> usize {
        self.glwe_rotate_tmp_bytes()
    }

    fn ggsw_rotate<'r, 'a>(&self, k: i64, res: &mut GGSWBackendMut<'r, BE>, a: &GGSWBackendRef<'a, BE>) {
        assert!(res.dnum() <= a.dnum());
        assert_eq!(res.dsize(), a.dsize());
        assert_eq!(res.rank(), a.rank());
        let rows: usize = res.dnum().into();
        let cols: usize = (res.rank() + 1).into();

        for row in 0..rows {
            for col in 0..cols {
                self.glwe_rotate(
                    k,
                    &mut ggsw_at_backend_mut_from_mut::<BE>(res, row, col),
                    &ggsw_at_backend_ref_from_ref::<BE>(a, row, col),
                );
            }
        }
    }

    fn ggsw_rotate_assign<'s, 'r>(&self, k: i64, res: &mut GGSWBackendMut<'r, BE>, scratch: &mut ScratchArena<'s, BE>)
    where
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE> + ScratchAvailable,
    {
        assert!(
            scratch.available() >= self.ggsw_rotate_tmp_bytes(),
            "scratch.available(): {} < GGSWRotate::ggsw_rotate_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_rotate_tmp_bytes()
        );

        let rows: usize = res.dnum().into();
        let cols: usize = (res.rank() + 1).into();

        let mut scratch = scratch.borrow();
        for row in 0..rows {
            for col in 0..cols {
                self.glwe_rotate_assign(k, &mut ggsw_at_backend_mut_from_mut::<BE>(res, row, col), &mut scratch);
            }
        }
    }
}

pub trait GLWEMulXpMinusOne<BE: Backend>
where
    Self: ModuleN + VecZnxMulXpMinusOneBackend<BE> + VecZnxMulXpMinusOneAssignBackend<BE>,
{
    fn glwe_mul_xp_minus_one<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        let res = &mut res.to_backend_mut();
        let a = &a.to_backend_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank());

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_mul_xp_minus_one_backend(k, &mut res.data, i, &a.data, i);
        }
    }

    fn glwe_mul_xp_minus_one_assign<'s, R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
    {
        let res = &mut res.to_backend_mut();

        assert_eq!(res.n(), self.n() as u32);

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_mul_xp_minus_one_assign_backend(k, &mut res.data, i, &mut scratch.borrow());
        }
    }
}

pub trait GLWECopy<BE: Backend>
where
    Self: ModuleN + VecZnxCopyBackend<BE> + VecZnxZeroBackend<BE>,
{
    fn glwe_copy(&self, res: &mut GLWEBackendMut<'_, BE>, a: &GLWEBackendRef<'_, BE>) {
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert!(res.rank() == a.rank() || a.rank() == 0);

        let min_rank: usize = res.rank().min(a.rank()).as_usize() + 1;

        for i in 0..min_rank {
            self.vec_znx_copy_backend(&mut res.data, i, &a.data, i);
        }

        for i in min_rank..(res.rank() + 1).into() {
            self.vec_znx_zero_backend(&mut res.data, i);
        }
    }
}

pub trait GLWEShift<BE: Backend>
where
    Self: ModuleN
        + VecZnxRshAssignBackend<BE>
        + VecZnxLshAddIntoBackend<BE>
        + VecZnxLshSubBackend<BE>
        + VecZnxRshTmpBytes
        + VecZnxLshTmpBytes
        + VecZnxLshAssignBackend<BE>
        + VecZnxLshBackend<BE>,
{
    fn glwe_shift_tmp_bytes(&self) -> usize {
        self.vec_znx_rsh_tmp_bytes().max(self.vec_znx_lsh_tmp_bytes())
    }

    fn glwe_rsh<'s, R>(&self, k: usize, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let res = &mut res.to_backend_mut();
        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );
        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_rsh_assign_backend(base2k, k, &mut res.data, i, &mut scratch.borrow());
        }
    }

    fn glwe_lsh_assign<'s, R>(&self, res: &mut R, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let res = &mut res.to_backend_mut();

        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_lsh_assign_backend(base2k, k, &mut res.data, i, &mut scratch.borrow());
        }
    }

    fn glwe_lsh<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let res = &mut res.to_backend_mut();
        let a = &a.to_backend_ref();
        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_lsh_backend(base2k, k, &mut res.data, i, &a.data, i, &mut scratch.borrow());
        }
    }

    fn glwe_lsh_add<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let res = &mut res.to_backend_mut();
        let a = &a.to_backend_ref();
        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_lsh_add_into_backend(base2k, k, &mut res.data, i, &a.data, i, &mut scratch.borrow());
        }
    }

    fn glwe_lsh_sub<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let res = &mut res.to_backend_mut();
        let a = &a.to_backend_ref();
        assert!(
            scratch.available() >= self.glwe_shift_tmp_bytes(),
            "scratch.available(): {} < GLWEShift::glwe_shift_tmp_bytes: {}",
            scratch.available(),
            self.glwe_shift_tmp_bytes()
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_lsh_sub_backend(base2k, k, &mut res.data, i, &a.data, i, &mut scratch.borrow());
        }
    }
}

pub trait GLWENormalize<BE: Backend>
where
    Self: ModuleN + VecZnxNormalize<BE> + VecZnxNormalizeAssignBackend<BE> + VecZnxNormalizeTmpBytes,
{
    fn glwe_normalize_tmp_bytes(&self) -> usize {
        self.vec_znx_normalize_tmp_bytes()
    }

    fn glwe_maybe_cross_normalize_to_ref<'a>(
        &self,
        glwe: &'a GLWEBackendRef<'a, BE>,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWEBackendMut<'a, BE>>,
        scratch: &'a mut ScratchArena<'a, BE>,
    ) -> GLWEBackendRef<'a, BE>
    where
        ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        if glwe.base2k().as_usize() == target_base2k {
            tmp_slot.take();
            return glwe_backend_ref_from_ref::<BE>(glwe);
        }

        let mut layout = glwe.glwe_layout();
        layout.base2k = target_base2k.into();

        let (tmp, mut scratch2) = scratch.borrow().take_glwe(&layout);
        *tmp_slot = Some(tmp);

        let tmp_ref = tmp_slot.as_mut().expect("tmp_slot just set to Some, but found None");
        let glwe_ref = glwe_backend_ref_from_ref::<BE>(glwe);
        self.glwe_normalize(tmp_ref, &glwe_ref, &mut scratch2);

        glwe_backend_ref_from_mut::<BE>(tmp_ref)
    }

    fn glwe_maybe_cross_normalize_to_mut<'a>(
        &self,
        glwe: &'a mut GLWEBackendMut<'a, BE>,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWEBackendMut<'a, BE>>,
        scratch: &'a mut ScratchArena<'a, BE>,
    ) -> GLWEBackendMut<'a, BE>
    where
        ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        if glwe.base2k().as_usize() == target_base2k {
            tmp_slot.take();
            return glwe_backend_mut_from_mut::<BE>(glwe);
        }

        let mut layout = glwe.glwe_layout();
        layout.base2k = target_base2k.into();

        let (tmp, mut scratch2) = scratch.borrow().take_glwe(&layout);
        *tmp_slot = Some(tmp);

        let tmp_ref = tmp_slot.as_mut().expect("tmp_slot just set to Some, but found None");

        self.glwe_normalize(tmp_ref, &glwe_backend_ref_from_mut::<BE>(&*glwe), &mut scratch2);

        glwe_backend_mut_from_mut::<BE>(tmp_ref)
    }

    fn glwe_normalize<'s, 'r, 'a>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank());
        assert!(
            scratch.available() >= self.glwe_normalize_tmp_bytes(),
            "scratch.available(): {} < GLWENormalize::glwe_normalize_tmp_bytes: {}",
            scratch.available(),
            self.glwe_normalize_tmp_bytes()
        );

        let res_base2k = res.base2k().into();

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_normalize(
                &mut res.data,
                res_base2k,
                0,
                i,
                &a.data,
                a.base2k().into(),
                i,
                &mut scratch.borrow(),
            );
        }
    }

    fn glwe_normalize_assign<'s, 'r>(&self, res: &mut GLWEBackendMut<'r, BE>, scratch: &mut ScratchArena<'s, BE>)
    where
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        assert!(
            scratch.available() >= self.glwe_normalize_tmp_bytes(),
            "scratch.available(): {} < GLWENormalize::glwe_normalize_tmp_bytes: {}",
            scratch.available(),
            self.glwe_normalize_tmp_bytes()
        );
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_normalize_assign_backend(res.base2k().into(), &mut res.data, i, &mut scratch.borrow());
        }
    }
}
