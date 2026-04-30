use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxAddAssignBackend, VecZnxAddIntoBackend,
        VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyTmpA, VecZnxLshAddIntoBackend, VecZnxLshBackend,
        VecZnxLshAssignBackend, VecZnxLshSubBackend, VecZnxLshTmpBytes, VecZnxMulXpMinusOneBackend,
        VecZnxMulXpMinusOneAssignBackend, VecZnxNegateBackend, VecZnxNegateAssignBackend, VecZnxNormalize,
        VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes, VecZnxRotateBackend, VecZnxRotateAssignBackend,
        VecZnxRotateAssignTmpBytes, VecZnxRshAssignBackend, VecZnxRshTmpBytes, VecZnxSubBackend, VecZnxSubAssignBackend,
        VecZnxSubNegateAssignBackend, VecZnxZeroBackend,
    },
    layouts::{
        Backend, CnvPVecLReborrowBackendRef, CnvPVecRReborrowBackendRef, Data, Module, ScratchArena, VecZnx,
        VecZnxBigReborrowBackendMut, VecZnxBigReborrowBackendRef, VecZnxDftReborrowBackendMut, VecZnxDftReborrowBackendRef,
        VecZnxReborrowBackendMut, VecZnxReborrowBackendRef,
    },
};

pub use crate::api::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWEMulXpMinusOne, GLWENegate, GLWENormalize, GLWERotate, GLWEShift, GLWESub,
    GLWETensoring,
};
use crate::{
    GGLWEProduct, ScratchArenaTakeCore,
    layouts::{
        Base2K, GGLWEInfos, GLWE, GLWEBackendMut, GLWEBackendRef, GLWEInfos, GLWEPlaintext, GLWEPlaintextToBackendRef,
        GLWETensor, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, glwe_backend_mut_from_mut,
        glwe_backend_ref_from_mut, glwe_backend_ref_from_ref, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};

#[doc(hidden)]
pub trait GLWEMulConstDefault<BE: Backend> {
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

impl<BE: Backend> GLWEMulConstDefault<BE> for Module<BE>
where
    Self: Convolution<BE> + VecZnxBigBytesOf + VecZnxBigNormalize<BE> + VecZnxBigNormalizeTmpBytes,
    Self: VecZnxCopyBackend<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_mul_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b_size: usize) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        assert_eq!(self.n() as u32, res.n());
        assert_eq!(self.n() as u32, a.n());

        let a_base2k: usize = a.base2k().as_usize();
        let res_base2k: usize = res.base2k().as_usize();
        let cnv_offset = a.size().max(b_size);
        let res_size: usize = (res.size() * res_base2k).div_ceil(a_base2k);
        let res_dft_size: usize = a.size() + b_size - cnv_offset.saturating_sub(1);
        let lvl_0: usize = self.bytes_of_vec_znx_big(1, res_dft_size) + VecZnx::bytes_of(self.n(), 1, res.size());
        let lvl_1_cnv: usize = self.cnv_by_const_apply_tmp_bytes(res_size, cnv_offset, a.size(), b_size);
        let lvl_1_norm: usize = self.vec_znx_big_normalize_tmp_bytes();
        let lvl_1: usize = lvl_1_cnv.max(lvl_1_norm);

        lvl_0 + lvl_1
    }

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
        GLWE<A>: GLWEToBackendRef<BE>,
    {
        let scratch = scratch.borrow();
        assert_eq!(res.rank(), a.rank());
        assert!(
            scratch.available() >= self.glwe_mul_const_tmp_bytes(res, a, b.len()),
            "scratch.available(): {} < GLWEMulConst::glwe_mul_const_tmp_bytes: {}",
            scratch.available(),
            self.glwe_mul_const_tmp_bytes(res, a, b.len())
        );

        let cols: usize = res.rank().as_usize() + 1;
        let a_base2k: usize = a.base2k().as_usize();
        let res_base2k: usize = res.base2k().as_usize();
        let a_backend = <GLWE<A> as GLWEToBackendRef<BE>>::to_backend_ref(a);

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < a_base2k {
            (0, -((a_base2k - (cnv_offset % a_base2k)) as i64))
        } else {
            ((cnv_offset / a_base2k).saturating_sub(1), (cnv_offset % a_base2k) as i64)
        };

        let res_dft_size = a.size() + b.len() - cnv_offset_hi;

        let (mut res_big, scratch) = scratch.take_vec_znx_big(self, 1, res_dft_size);
        let (mut res_tmp, mut scratch) = scratch.take_vec_znx(self.n(), 1, res.size());
        for i in 0..cols {
            {
                let mut scratch_iter = scratch.borrow();
                let mut res_big_backend = res_big.reborrow_backend_mut();
                self.cnv_by_const_apply(
                    cnv_offset_hi,
                    &mut res_big_backend,
                    0,
                    &a_backend.data,
                    i,
                    b,
                    &mut scratch_iter,
                );
            }
            let res_big_ref = res_big.reborrow_backend_ref();
            {
                let mut scratch_iter = scratch.borrow();
                self.vec_znx_big_normalize(
                    &mut res_tmp,
                    res_base2k,
                    cnv_offset_lo,
                    0,
                    &res_big_ref,
                    a_base2k,
                    0,
                    &mut scratch_iter,
                );
            }
            let mut res_backend = <GLWE<R> as GLWEToBackendMut<BE>>::to_backend_mut(res);
            let res_tmp_ref = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&res_tmp);
            self.vec_znx_copy_backend(&mut res_backend.data, i, &res_tmp_ref, 0);
        }
    }

    fn glwe_mul_const_assign<'s, R>(&self, cnv_offset: usize, res: &mut GLWE<R>, b: &[i64], scratch: &mut ScratchArena<'s, BE>)
    where
        R: Data,
        GLWE<R>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    {
        let scratch = scratch.borrow();
        assert!(
            scratch.available() >= self.glwe_mul_const_tmp_bytes(res, res, b.len()),
            "scratch.available(): {} < GLWEMulConst::glwe_mul_const_tmp_bytes: {}",
            scratch.available(),
            self.glwe_mul_const_tmp_bytes(res, res, b.len())
        );

        let cols: usize = res.rank().as_usize() + 1;
        let res_base2k: usize = res.base2k().as_usize();

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < res_base2k {
            (0, -((res_base2k - (cnv_offset % res_base2k)) as i64))
        } else {
            ((cnv_offset / res_base2k).saturating_sub(1), (cnv_offset % res_base2k) as i64)
        };

        let (mut res_big, scratch) = scratch.take_vec_znx_big(self, 1, res.size());
        let (mut res_tmp, mut scratch) = scratch.take_vec_znx(self.n(), 1, res.size());
        for i in 0..cols {
            {
                let res_backend = <GLWE<R> as GLWEToBackendRef<BE>>::to_backend_ref(res);
                let mut scratch_iter = scratch.borrow();
                let mut res_big_backend = res_big.reborrow_backend_mut();
                self.cnv_by_const_apply(
                    cnv_offset_hi,
                    &mut res_big_backend,
                    0,
                    &res_backend.data,
                    i,
                    b,
                    &mut scratch_iter,
                );
            }
            let res_big_ref = res_big.reborrow_backend_ref();
            {
                let mut scratch_iter = scratch.borrow();
                self.vec_znx_big_normalize(
                    &mut res_tmp,
                    res_base2k,
                    cnv_offset_lo,
                    0,
                    &res_big_ref,
                    res_base2k,
                    0,
                    &mut scratch_iter,
                );
            }
            let mut res_backend = <GLWE<R> as GLWEToBackendMut<BE>>::to_backend_mut(res);
            let res_tmp_ref = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&res_tmp);
            self.vec_znx_copy_backend(&mut res_backend.data, i, &res_tmp_ref, 0);
        }
    }
}

impl<BE: Backend> GLWEMulPlainDefault<BE> for Module<BE>
where
    Self: Sized
        + ModuleN
        + CnvPVecBytesOf
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + Convolution<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopyBackend<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_mul_plain_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        assert_eq!(self.n() as u32, res.n());
        assert_eq!(self.n() as u32, a.n());
        assert_eq!(self.n() as u32, b.n());

        let ab_base2k: Base2K = a.base2k();
        assert_eq!(b.base2k(), ab_base2k);

        let cols: usize = res.rank().as_usize() + 1;

        let a_size: usize = a.size();
        let b_size: usize = b.size();
        let cnv_offset: usize = a_size.min(b_size);

        let lvl_0: usize = self.bytes_of_cnv_pvec_left(cols, a_size) + self.bytes_of_cnv_pvec_right(1, b_size);
        let lvl_1: usize = self
            .cnv_prepare_left_tmp_bytes(a_size, a_size)
            .max(self.cnv_prepare_right_tmp_bytes(b_size, b_size));

        let res_dft_size =
            normalize_input_limb_bound_worst_case(a_size + b_size, res.size(), res.base2k().as_usize(), ab_base2k.as_usize());
        let lvl_2_cnv_apply: usize = self.cnv_apply_dft_tmp_bytes(res_dft_size, cnv_offset, a_size, b_size);

        let lvl_2_res_dft: usize = self.bytes_of_vec_znx_dft(1, res_dft_size);
        let lvl_2_res_tmp: usize = self.bytes_of_vec_znx_big(1, res_dft_size) + VecZnx::bytes_of(self.n(), 1, res.size());
        let lvl_2_norm: usize = self.vec_znx_big_normalize_tmp_bytes();
        let lvl_2: usize = lvl_2_res_tmp + lvl_2_res_dft + lvl_2_cnv_apply.max(lvl_2_norm);

        lvl_0 + lvl_1.max(lvl_2)
    }

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
        GLWEPlaintext<B>: GLWEPlaintextToBackendRef<BE>,
    {
        let scratch = scratch.borrow();
        assert_eq!(res.rank(), a.rank());
        assert!(
            scratch.available() >= self.glwe_mul_plain_tmp_bytes(res, a, b),
            "scratch.available(): {} < GLWEMulPlain::glwe_mul_plain_tmp_bytes: {}",
            scratch.available(),
            self.glwe_mul_plain_tmp_bytes(res, a, b)
        );

        let ab_base2k: usize = a.base2k().as_usize();
        assert_eq!(b.base2k().as_usize(), ab_base2k);
        assert_eq!(a_effective_k.div_ceil(ab_base2k), a.size());
        assert_eq!(b_effective_k.div_ceil(ab_base2k), b.size());
        let res_base2k: usize = res.base2k().as_usize();

        let cols: usize = res.rank().as_usize() + 1;

        let (mut a_prep, scratch) = scratch.take_cnv_pvec_left(self, cols, a.size());
        let (mut b_prep, mut scratch) = scratch.take_cnv_pvec_right(self, 1, b.size());

        let a_mask = msb_mask_bottom_limb(ab_base2k, a_effective_k);
        let b_mask = msb_mask_bottom_limb(ab_base2k, b_effective_k);
        let a_backend = <GLWE<A> as GLWEToBackendRef<BE>>::to_backend_ref(a);
        let b_backend = <GLWEPlaintext<B> as GLWEPlaintextToBackendRef<BE>>::to_backend_ref(b);

        scratch = scratch.apply_mut(|scratch| self.cnv_prepare_left(&mut a_prep, &a_backend.data, a_mask, scratch));
        scratch = scratch.apply_mut(|scratch| self.cnv_prepare_right(&mut b_prep, &b_backend.data, b_mask, scratch));

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < ab_base2k {
            (0, -((ab_base2k - (cnv_offset % ab_base2k)) as i64))
        } else {
            ((cnv_offset / ab_base2k).saturating_sub(1), (cnv_offset % ab_base2k) as i64)
        };

        let res_dft_size = a.size() + b.size() - cnv_offset_hi;
        let (mut res_tmp, mut scratch) = scratch.take_vec_znx(self.n(), 1, res.size());

        for i in 0..cols {
            let (mut res_dft, mut scratch_3) = scratch.borrow().take_vec_znx_dft(self, 1, res_dft_size);
            {
                let mut res_dft_backend = res_dft.reborrow_backend_mut();
                self.cnv_apply_dft(
                    cnv_offset_hi,
                    &mut res_dft_backend,
                    0,
                    &a_prep.reborrow_backend_ref(),
                    i,
                    &b_prep.reborrow_backend_ref(),
                    0,
                    &mut scratch_3,
                );
            }
            let (mut res_big, mut scratch_4) = scratch_3.take_vec_znx_big(self, 1, res_dft_size);
            {
                let mut res_big_backend = res_big.reborrow_backend_mut();
                let mut res_dft_backend = res_dft.reborrow_backend_mut();
                self.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
            }
            let res_big_ref = res_big.reborrow_backend_ref();
            {
                let mut scratch_iter = scratch_4.borrow();
                self.vec_znx_big_normalize(
                    &mut res_tmp,
                    res_base2k,
                    cnv_offset_lo,
                    0,
                    &res_big_ref,
                    ab_base2k,
                    0,
                    &mut scratch_iter,
                );
            }
            let mut res_backend = <GLWE<R> as GLWEToBackendMut<BE>>::to_backend_mut(res);
            let res_tmp_ref = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&res_tmp);
            self.vec_znx_copy_backend(&mut res_backend.data, i, &res_tmp_ref, 0);
        }
    }

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
        GLWEPlaintext<A>: GLWEPlaintextToBackendRef<BE>,
    {
        let scratch = scratch.borrow();
        assert!(
            scratch.available() >= self.glwe_mul_plain_tmp_bytes(res, res, a),
            "scratch.available(): {} < GLWEMulPlain::glwe_mul_plain_tmp_bytes: {}",
            scratch.available(),
            self.glwe_mul_plain_tmp_bytes(res, res, a)
        );

        let ab_base2k: usize = a.base2k().as_usize();
        assert_eq!(res.base2k().as_usize(), ab_base2k);
        assert_eq!(res_effective_k.div_ceil(ab_base2k), res.size());
        assert_eq!(a_effective_k.div_ceil(ab_base2k), a.size());

        let cols: usize = res.rank().as_usize() + 1;

        let (mut res_prep, scratch) = scratch.take_cnv_pvec_left(self, cols, res.size());
        let (mut a_prep, mut scratch) = scratch.take_cnv_pvec_right(self, 1, a.size());

        let mask_res = msb_mask_bottom_limb(ab_base2k, res_effective_k);
        let mask_a = msb_mask_bottom_limb(ab_base2k, a_effective_k);
        let a_backend = <GLWEPlaintext<A> as GLWEPlaintextToBackendRef<BE>>::to_backend_ref(a);

        scratch = scratch.apply_mut(|scratch| {
            let res_backend = <GLWE<R> as GLWEToBackendRef<BE>>::to_backend_ref(res);
            self.cnv_prepare_left(&mut res_prep, &res_backend.data, mask_res, scratch)
        });
        scratch = scratch.apply_mut(|scratch| self.cnv_prepare_right(&mut a_prep, &a_backend.data, mask_a, scratch));

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < ab_base2k {
            (0, -((ab_base2k - (cnv_offset % ab_base2k)) as i64))
        } else {
            ((cnv_offset / ab_base2k).saturating_sub(1), (cnv_offset % ab_base2k) as i64)
        };

        let res_dft_size = a.size() + res.size() - cnv_offset_hi;
        let (mut res_tmp, mut scratch) = scratch.take_vec_znx(self.n(), 1, res.size());

        for i in 0..cols {
            let (mut res_dft, mut scratch_3) = scratch.borrow().take_vec_znx_dft(self, 1, res_dft_size);
            {
                let mut res_dft_backend = res_dft.reborrow_backend_mut();
                self.cnv_apply_dft(
                    cnv_offset_hi,
                    &mut res_dft_backend,
                    0,
                    &res_prep.reborrow_backend_ref(),
                    i,
                    &a_prep.reborrow_backend_ref(),
                    0,
                    &mut scratch_3,
                );
            }
            let (mut res_big, mut scratch_4) = scratch_3.take_vec_znx_big(self, 1, res_dft_size);
            {
                let mut res_big_backend = res_big.reborrow_backend_mut();
                let mut res_dft_backend = res_dft.reborrow_backend_mut();
                self.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
            }
            let res_big_ref = res_big.reborrow_backend_ref();
            {
                let mut scratch_iter = scratch_4.borrow();
                self.vec_znx_big_normalize(
                    &mut res_tmp,
                    ab_base2k,
                    cnv_offset_lo,
                    0,
                    &res_big_ref,
                    ab_base2k,
                    0,
                    &mut scratch_iter,
                );
            }
            let mut res_backend = <GLWE<R> as GLWEToBackendMut<BE>>::to_backend_mut(res);
            let res_tmp_ref = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&res_tmp);
            self.vec_znx_copy_backend(&mut res_backend.data, i, &res_tmp_ref, 0);
        }
    }
}

#[doc(hidden)]
pub trait GLWEMulPlainDefault<BE: Backend> {
    fn glwe_mul_plain_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWEPlaintext<B>,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWE<A>: GLWEToBackendRef<BE>,
        GLWEPlaintext<B>: GLWEPlaintextToBackendRef<BE>;

    fn glwe_mul_plain_assign<R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        res_effective_k: usize,
        a: &GLWEPlaintext<A>,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: Data,
        A: Data,
        GLWE<R>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        GLWEPlaintext<A>: GLWEPlaintextToBackendRef<BE>;
}

#[doc(hidden)]
pub trait GLWETensoringDefault<BE: Backend> {
    fn glwe_tensor_square_apply_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    fn glwe_tensor_apply_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(&self, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    fn glwe_tensor_relinearize<R, A, B>(
        &self,
        res: &mut GLWE<R>,
        a: &GLWETensor<A>,
        tsk: &GLWETensorKeyPrepared<B, BE>,
        tsk_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWETensorKeyPrepared<B, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
        GLWETensor<A>: GLWEToBackendRef<BE>;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_square_apply<R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: Data,
        A: Data,
        GLWETensor<R>: GLWEToBackendMut<BE>,
        GLWE<A>: GLWEToBackendRef<BE>;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWETensor<R>: GLWEToBackendMut<BE>,
        GLWE<A>: GLWEToBackendRef<BE>,
        GLWE<B>: GLWEToBackendRef<BE>;
}

impl<BE: Backend> GLWETensoringDefault<BE> for Module<BE>
where
    Self: Sized
        + ModuleN
        + CnvPVecBytesOf
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + Convolution<BE>
        + VecZnxSubAssignBackend<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxNegateBackend<BE>
        + GGLWEProduct<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxNormalizeTmpBytes,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_tensor_square_apply_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        assert_eq!(self.n() as u32, res.n());
        assert_eq!(self.n() as u32, a.n());

        let cols: usize = res.rank().as_usize() + 1;
        let a_size: usize = a.size();
        let res_size: usize = res.size();
        let cnv_offset = a_size;

        let lvl_0: usize = self.bytes_of_cnv_pvec_left(cols, a_size) + self.bytes_of_cnv_pvec_right(cols, a_size);
        let lvl_diag_cache: usize = VecZnx::bytes_of(self.n(), cols, res_size);
        let lvl_1: usize = self.cnv_prepare_self_tmp_bytes(a_size, a_size);
        let diag_dft_size =
            normalize_input_limb_bound_worst_case(2 * a_size, res_size, res.base2k().as_usize(), a.base2k().as_usize());
        let lvl_2_apply: usize = self.cnv_apply_dft_tmp_bytes(diag_dft_size, cnv_offset, a_size, a_size);
        let pairwise_dft_size =
            normalize_input_limb_bound_worst_case(2 * a_size, res_size, res.base2k().as_usize(), a.base2k().as_usize());
        let lvl_2_pairwise: usize = self.cnv_pairwise_apply_dft_tmp_bytes(cnv_offset, pairwise_dft_size, a_size, a_size);

        let lvl_2a: usize = self.bytes_of_vec_znx_dft(1, diag_dft_size)
            + self.bytes_of_vec_znx_big(1, diag_dft_size)
            + VecZnx::bytes_of(self.n(), 1, res_size)
            + lvl_2_apply.max(self.vec_znx_big_normalize_tmp_bytes());
        let lvl_2b: usize = self.bytes_of_vec_znx_dft(1, pairwise_dft_size)
            + self.bytes_of_vec_znx_big(1, pairwise_dft_size)
            + VecZnx::bytes_of(self.n(), 1, res_size)
            + lvl_2_pairwise.max(self.vec_znx_big_normalize_tmp_bytes());
        let lvl_2: usize = lvl_2a.max(lvl_2b);

        lvl_0 + lvl_diag_cache + lvl_1.max(lvl_2)
    }

    fn glwe_tensor_apply_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        assert_eq!(self.n() as u32, res.n());
        assert_eq!(self.n() as u32, a.n());
        assert_eq!(self.n() as u32, b.n());

        let ab_base2k: Base2K = a.base2k();
        assert_eq!(b.base2k(), ab_base2k);

        let cols: usize = res.rank().as_usize() + 1;

        let a_size: usize = a.size();
        let b_size: usize = b.size();
        let res_size: usize = res.size();
        let cnv_offset = a_size.min(b_size);

        let lvl_0: usize = self.bytes_of_cnv_pvec_left(cols, a_size) + self.bytes_of_cnv_pvec_right(cols, b_size);
        let lvl_1: usize = self
            .cnv_prepare_left_tmp_bytes(a_size, a_size)
            .max(self.cnv_prepare_right_tmp_bytes(b_size, b_size));
        let diag_dft_size =
            normalize_input_limb_bound_worst_case(a_size + b_size, res_size, res.base2k().as_usize(), ab_base2k.as_usize());
        let lvl_2_apply: usize = self.cnv_apply_dft_tmp_bytes(diag_dft_size, cnv_offset, a_size, b_size);
        let pairwise_dft_size =
            normalize_input_limb_bound_worst_case(a_size + b_size, res_size, res.base2k().as_usize(), ab_base2k.as_usize());
        let lvl_2_pairwise: usize = self.cnv_pairwise_apply_dft_tmp_bytes(cnv_offset, pairwise_dft_size, a_size, b_size);

        let lvl_2a: usize = self.bytes_of_vec_znx_dft(1, diag_dft_size)
            + self.bytes_of_vec_znx_big(1, diag_dft_size)
            + VecZnx::bytes_of(self.n(), 1, res_size)
            + lvl_2_apply.max(self.vec_znx_big_normalize_tmp_bytes());
        let lvl_2b: usize = self.bytes_of_vec_znx_dft(1, pairwise_dft_size)
            + self.bytes_of_vec_znx_big(1, pairwise_dft_size)
            + VecZnx::bytes_of(self.n(), 1, res_size)
            + lvl_2_pairwise.max(self.vec_znx_big_normalize_tmp_bytes());
        let lvl_2: usize = lvl_2a.max(lvl_2b);

        lvl_0 + lvl_1.max(lvl_2)
    }

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(&self, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, res.n());
        assert_eq!(self.n() as u32, a.n());
        assert_eq!(self.n() as u32, tsk.n());

        let a_base2k: usize = a.base2k().into();
        let key_base2k: usize = tsk.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let cols: usize = tsk.rank_out().as_usize() + 1;
        let pairs: usize = tsk.rank_in().as_usize();

        let a_dft_size: usize = (a.size() * a_base2k).div_ceil(key_base2k);

        let lvl_0: usize = self.bytes_of_vec_znx_dft(pairs, a_dft_size);

        let lvl_1_pre_conv: usize = if a_base2k != key_base2k {
            VecZnx::bytes_of(self.n(), 1, a_dft_size) + self.vec_znx_normalize_tmp_bytes()
        } else {
            0
        };
        let lvl_1_res_dft: usize = self.bytes_of_vec_znx_dft(cols, tsk.size());
        let lvl_1_gglwe_product: usize = self.gglwe_product_dft_tmp_bytes(res.size(), a_dft_size, tsk);
        let lvl_1_post_conv: usize = if res_base2k != key_base2k {
            VecZnx::bytes_of(self.n(), 1, a_dft_size) + self.vec_znx_normalize_tmp_bytes()
        } else {
            0
        };
        let lvl_1_big_norm: usize = self.bytes_of_vec_znx_big(cols, tsk.size())
            + VecZnx::bytes_of(self.n(), 1, res.size())
            + self.vec_znx_big_normalize_tmp_bytes();
        let lvl_1_main: usize = lvl_1_res_dft + lvl_1_gglwe_product.max(lvl_1_post_conv).max(lvl_1_big_norm);
        let lvl_1: usize = lvl_1_pre_conv.max(lvl_1_main);

        lvl_0 + lvl_1
    }

    fn glwe_tensor_relinearize<R, A, B>(
        &self,
        res: &mut GLWE<R>,
        a: &GLWETensor<A>,
        tsk: &GLWETensorKeyPrepared<B, BE>,
        tsk_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWETensorKeyPrepared<B, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
        GLWETensor<A>: GLWEToBackendRef<BE>,
    {
        let scratch = scratch.borrow();
        assert!(
            scratch.available() >= self.glwe_tensor_relinearize_tmp_bytes(res, a, tsk),
            "scratch.available(): {} < GLWETensoring::glwe_tensor_relinearize_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_relinearize_tmp_bytes(res, a, tsk)
        );

        let a_base2k: usize = a.base2k().into();
        let key_base2k: usize = tsk.base2k().into();
        let res_base2k: usize = res.base2k().into();
        let a_backend = a.to_backend_ref();

        assert_eq!(res.rank(), tsk.rank_out());
        assert_eq!(a.rank(), tsk.rank_out());

        let cols: usize = tsk.rank_out().as_usize() + 1;
        let pairs: usize = tsk.rank_in().as_usize();

        let a_dft_size: usize = (a.size() * a_base2k).div_ceil(key_base2k);

        let (mut a_dft, mut scratch) = scratch.take_vec_znx_dft(self, pairs, a_dft_size);

        {
            let (mut a_conv, mut scratch_norm) = scratch.borrow().take_vec_znx(self.n(), 1, a_dft_size);
            for i in 0..pairs {
                let mut scratch_iter = scratch_norm.borrow();
                self.vec_znx_normalize(
                    &mut a_conv,
                    key_base2k,
                    0,
                    0,
                    &a_backend.data,
                    a_base2k,
                    cols + i,
                    &mut scratch_iter,
                );
                let a_conv_ref = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&a_conv);
                self.vec_znx_dft_apply(1, 0, &mut a_dft, i, &a_conv_ref, 0);
            }
        }

        let (mut res_dft, mut scratch_2) = scratch.borrow().take_vec_znx_dft(self, cols, tsk_size); // Todo optimise
        let tsk = tsk.to_backend_ref();

        let a_dft_ref =
            <poulpy_hal::layouts::VecZnxDft<BE::BufMut<'_>, BE> as VecZnxDftReborrowBackendRef<BE>>::reborrow_backend_ref(&a_dft);
        self.gglwe_product_dft(&mut res_dft, &a_dft_ref, &tsk.0, &mut scratch_2);
        let (mut res_big, mut scratch_3) = scratch_2.take_vec_znx_big(self, cols, tsk_size);
        {
            let mut res_big_backend = res_big.reborrow_backend_mut();
            let mut res_dft_backend = res_dft.reborrow_backend_mut();
            for i in 0..cols {
                self.vec_znx_idft_apply_tmpa(&mut res_big_backend, i, &mut res_dft_backend, i);
            }
        }

        {
            let (mut a_conv, mut scratch_norm) = scratch_3.borrow().take_vec_znx(self.n(), 1, a_dft_size);
            for i in 0..cols {
                let mut scratch_iter = scratch_norm.borrow();
                self.vec_znx_normalize(&mut a_conv, key_base2k, 0, 0, &a_backend.data, a_base2k, i, &mut scratch_iter);
                let a_conv_ref = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&a_conv);
                self.vec_znx_big_add_small_assign(&mut res_big, i, &a_conv_ref, 0);
            }
        }

        {
            let (mut res_tmp, mut scratch_norm) = scratch_3.borrow().take_vec_znx(self.n(), 1, res.size());
            for i in 0..(res.rank() + 1).into() {
                let res_big_ref =
                    <poulpy_hal::layouts::VecZnxBig<BE::BufMut<'_>, BE> as VecZnxBigReborrowBackendRef<BE>>::reborrow_backend_ref(
                        &res_big,
                    );
                let mut scratch_iter = scratch_norm.borrow();
                self.vec_znx_big_normalize(&mut res_tmp, res_base2k, 0, 0, &res_big_ref, key_base2k, i, &mut scratch_iter);
                let mut res_backend = <GLWE<R> as GLWEToBackendMut<BE>>::to_backend_mut(res);
                let res_tmp_ref = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&res_tmp);
                self.vec_znx_copy_backend(&mut res_backend.data, i, &res_tmp_ref, 0);
            }
        }
    }

    fn glwe_tensor_square_apply<R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: Data,
        A: Data,
        GLWETensor<R>: GLWEToBackendMut<BE>,
        GLWE<A>: GLWEToBackendRef<BE>,
    {
        let scratch = scratch.borrow();
        assert!(
            scratch.available() >= self.glwe_tensor_square_apply_tmp_bytes(res, a),
            "scratch.available(): {} < GLWETensoring::glwe_tensor_square_apply_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_square_apply_tmp_bytes(res, a)
        );

        let a_base2k: usize = a.base2k().as_usize();

        assert_eq!(a_effective_k.div_ceil(a_base2k), a.size());

        let res_base2k: usize = res.base2k().as_usize();
        let cols: usize = res.rank().as_usize() + 1;

        let (mut a_prep, scratch) = scratch.take_cnv_pvec_left(self, cols, a.size());
        let (mut b_prep, mut scratch) = scratch.take_cnv_pvec_right(self, cols, a.size());

        let a_mask = msb_mask_bottom_limb(a_base2k, a_effective_k);
        let a_backend = <GLWE<A> as GLWEToBackendRef<BE>>::to_backend_ref(a);

        let mut prep_scratch = scratch.borrow();
        self.cnv_prepare_self(&mut a_prep, &mut b_prep, &a_backend.data, a_mask, &mut prep_scratch);
        let (mut diag_terms, mut scratch) = scratch.take_vec_znx(self.n(), cols, res.size());

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < a_base2k {
            (0, -((a_base2k - (cnv_offset % a_base2k)) as i64))
        } else {
            ((cnv_offset / a_base2k).saturating_sub(1), (cnv_offset % a_base2k) as i64)
        };

        let diag_dft_size =
            normalize_input_limb_bound_with_offset(2 * a.size() - cnv_offset_hi, res.size(), res_base2k, a_base2k, cnv_offset_lo);
        let pairwise_dft_size =
            normalize_input_limb_bound_with_offset(2 * a.size() - cnv_offset_hi, res.size(), res_base2k, a_base2k, cnv_offset_lo);

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            let (mut res_dft, mut scratch_4) = scratch.borrow().take_vec_znx_dft(self, 1, diag_dft_size);
            {
                let mut res_dft_backend = res_dft.reborrow_backend_mut();
                self.cnv_apply_dft(
                    cnv_offset_hi,
                    &mut res_dft_backend,
                    0,
                    &a_prep.reborrow_backend_ref(),
                    i,
                    &b_prep.reborrow_backend_ref(),
                    i,
                    &mut scratch_4,
                );
            }
            let (mut res_big, scratch_5) = scratch_4.take_vec_znx_big(self, 1, diag_dft_size);
            {
                let mut res_big_backend = res_big.reborrow_backend_mut();
                let mut res_dft_backend = res_dft.reborrow_backend_mut();
                self.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
            }
            let (mut tmp, mut scratch_6) = scratch_5.take_vec_znx(self.n(), 1, res.size());
            let res_big_ref = res_big.reborrow_backend_ref();
            let mut scratch_iter = scratch_6.borrow();
            self.vec_znx_big_normalize(
                &mut tmp,
                res_base2k,
                cnv_offset_lo,
                0,
                &res_big_ref,
                a_base2k,
                0,
                &mut scratch_iter,
            );

            // TODO: Do we need 2 copies?
            {
                let mut diag_terms_mut =
                    <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendMut<BE>>::reborrow_backend_mut(&mut diag_terms);
                let tmp_ref = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&tmp);
                self.vec_znx_copy_backend(&mut diag_terms_mut, i, &tmp_ref, 0);
            }
            {
                let mut res_backend = <GLWETensor<R> as GLWEToBackendMut<BE>>::to_backend_mut(res);
                let diag_terms_ref = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&diag_terms);
                self.vec_znx_copy_backend(&mut res_backend.data, col_i + i, &diag_terms_ref, i);
            }
        }

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            for j in i + 1..cols {
                let (mut res_dft, mut scratch_4) = scratch.borrow().take_vec_znx_dft(self, 1, pairwise_dft_size);
                {
                    let mut res_dft_backend = res_dft.reborrow_backend_mut();
                    self.cnv_pairwise_apply_dft(
                        cnv_offset_hi,
                        &mut res_dft_backend,
                        0,
                        &a_prep.reborrow_backend_ref(),
                        &b_prep.reborrow_backend_ref(),
                        i,
                        j,
                        &mut scratch_4,
                    );
                }
                let (mut res_big, scratch_6) = scratch_4.take_vec_znx_big(self, 1, pairwise_dft_size);
                {
                    let mut res_big_backend = res_big.reborrow_backend_mut();
                    let mut res_dft_backend = res_dft.reborrow_backend_mut();
                    self.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
                }
                let (mut tmp, mut scratch_7) = scratch_6.take_vec_znx(self.n(), 1, res.size());
                let res_big_ref = res_big.reborrow_backend_ref();
                let mut scratch_iter = scratch_7.borrow();
                self.vec_znx_big_normalize(
                    &mut tmp,
                    res_base2k,
                    cnv_offset_lo,
                    0,
                    &res_big_ref,
                    a_base2k,
                    0,
                    &mut scratch_iter,
                );
                {
                    let mut tmp_mut = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendMut<BE>>::reborrow_backend_mut(&mut tmp);
                    let diag_terms_ref =
                        <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&diag_terms);
                    self.vec_znx_sub_assign_backend(&mut tmp_mut, 0, &diag_terms_ref, i);
                    self.vec_znx_sub_assign_backend(&mut tmp_mut, 0, &diag_terms_ref, j);
                }

                // TODO: Do we need copy?
                let mut res_backend = <GLWETensor<R> as GLWEToBackendMut<BE>>::to_backend_mut(res);
                let tmp_ref = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&tmp);
                self.vec_znx_copy_backend(&mut res_backend.data, col_i + j, &tmp_ref, 0);
            }
        }
    }

    fn glwe_tensor_apply<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWETensor<R>: GLWEToBackendMut<BE>,
        GLWE<A>: GLWEToBackendRef<BE>,
        GLWE<B>: GLWEToBackendRef<BE>,
    {
        let scratch = scratch.borrow();
        assert!(
            scratch.available() >= self.glwe_tensor_apply_tmp_bytes(res, a, b),
            "scratch.available(): {} < GLWETensoring::glwe_tensor_apply_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_apply_tmp_bytes(res, a, b)
        );

        let ab_base2k: usize = a.base2k().as_usize();
        assert_eq!(b.base2k().as_usize(), ab_base2k);
        assert_eq!(a_effective_k.div_ceil(ab_base2k), a.size());
        assert_eq!(b_effective_k.div_ceil(ab_base2k), b.size());

        let res_base2k: usize = res.base2k().as_usize();

        let cols: usize = res.rank().as_usize() + 1;

        let (mut a_prep, scratch) = scratch.take_cnv_pvec_left(self, cols, a.size());
        let (mut b_prep, mut scratch) = scratch.take_cnv_pvec_right(self, cols, b.size());

        let a_mask = msb_mask_bottom_limb(ab_base2k, a_effective_k);
        let b_mask = msb_mask_bottom_limb(ab_base2k, b_effective_k);
        let a_backend = <GLWE<A> as GLWEToBackendRef<BE>>::to_backend_ref(a);
        let b_backend = <GLWE<B> as GLWEToBackendRef<BE>>::to_backend_ref(b);

        let mut prep_scratch = scratch.borrow();
        self.cnv_prepare_left(&mut a_prep, &a_backend.data, a_mask, &mut prep_scratch);
        self.cnv_prepare_right(&mut b_prep, &b_backend.data, b_mask, &mut prep_scratch);
        // Example for rank=3
        //
        // (a0, a1, a2, a3) x (b0, b1, b2, a3)
        //   L   L  L   L       R   R   R   R
        //
        // c(1)    = a0 * b0 				<- (L(a0) * R(b0))
        // c(s1)   = a0 * b1 + a1 * b0 		<- (L(a0) + L(a1)) * (R(b0) + R(b1)) + NEG(L(a0) * R(b0)) + SUB(L(a1) * R(b1))
        // c(s2)   = a0 * b2 + a2 * b0		<- (L(a0) + L(a2)) * (R(b0) + R(b2)) + NEG(L(a0) * R(b0)) + SUB(L(a2) * R(b2))
        // c(s3)   = a0 * b3 + a3 * b0		<- (L(a0) + L(a3)) * (R(b0) + R(b3)) + NEG(L(a0) * R(b0)) + SUB(L(a3) * R(b3))
        // c(s1^2) = a1 * b1 				<- (L(a1) * R(b1))
        // c(s1s2) = a1 * b2 + b2 * a1		<- (L(a1) + L(a2)) * (R(b1) + R(b2)) + NEG(L(a1) * R(b1)) + SUB(L(a2) * R(b2))
        // c(s1s3) = a1 * b3 + b3 * a1		<- (L(a1) + L(a3)) * (R(b1) + R(b3)) + NEG(L(a1) * R(b1)) + SUB(L(a3) * R(b3))
        // c(s2^2) = a2 * b2 				<- (L(a2) * R(b2))
        // c(s2s3) = a2 * b3 + a3 * b2 	    <- (L(a2) + L(a3)) * (R(b2) + R(b3)) + NEG(L(a2) * R(b2)) + SUB(L(a3) * R(b3))
        // c(s3^2) = a3 * b3				<- (L(a3) * R(b3))

        // Derive the offset. If cnv_offset < a_base2k, then we shift to a negative offset
        // since the convolution doesn't support negative offset (yet).
        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < ab_base2k {
            (0, -((ab_base2k - (cnv_offset % ab_base2k)) as i64))
        } else {
            ((cnv_offset / ab_base2k).saturating_sub(1), (cnv_offset % ab_base2k) as i64)
        };

        let diag_dft_size = normalize_input_limb_bound_with_offset(
            a.size() + b.size() - cnv_offset_hi,
            res.size(),
            res_base2k,
            ab_base2k,
            cnv_offset_lo,
        );
        let pairwise_dft_size = normalize_input_limb_bound_with_offset(
            a.size() + b.size() - cnv_offset_hi,
            res.size(),
            res_base2k,
            ab_base2k,
            cnv_offset_lo,
        );

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            let (mut res_dft, mut scratch_3) = scratch.borrow().take_vec_znx_dft(self, 1, diag_dft_size);
            {
                let mut res_dft_backend = res_dft.reborrow_backend_mut();
                self.cnv_apply_dft(
                    cnv_offset_hi,
                    &mut res_dft_backend,
                    0,
                    &a_prep.reborrow_backend_ref(),
                    i,
                    &b_prep.reborrow_backend_ref(),
                    i,
                    &mut scratch_3,
                );
            }
            let (mut res_big, scratch_4) = scratch_3.take_vec_znx_big(self, 1, diag_dft_size);
            {
                let mut res_big_backend = res_big.reborrow_backend_mut();
                let mut res_dft_backend = res_dft.reborrow_backend_mut();
                self.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
            }
            let (mut tmp, mut scratch_5) = scratch_4.take_vec_znx(self.n(), 1, res.size());
            let res_big_ref = res_big.reborrow_backend_ref();
            let mut scratch_iter = scratch_5.borrow();
            self.vec_znx_big_normalize(
                &mut tmp,
                res_base2k,
                cnv_offset_lo,
                0,
                &res_big_ref,
                ab_base2k,
                0,
                &mut scratch_iter,
            );

            {
                let mut res_backend = <GLWETensor<R> as GLWEToBackendMut<BE>>::to_backend_mut(res);
                let tmp_ref = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&tmp);
                self.vec_znx_copy_backend(&mut res_backend.data, col_i + i, &tmp_ref, 0);
            }

            // Pre-subtracts
            // res[i!=j] = NEG(a[i] * b[i]) + SUB(a[j] * b[j])
            for j in 0..cols {
                if j != i {
                    if j < i {
                        let col_j = j * cols - (j * (j + 1) / 2);
                        let mut res_backend = <GLWETensor<R> as GLWEToBackendMut<BE>>::to_backend_mut(res);
                        let tmp_ref = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&tmp);
                        self.vec_znx_sub_assign_backend(&mut res_backend.data, col_j + i, &tmp_ref, 0);
                    } else {
                        let mut res_backend = <GLWETensor<R> as GLWEToBackendMut<BE>>::to_backend_mut(res);
                        let tmp_ref = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&tmp);
                        self.vec_znx_negate_backend(&mut res_backend.data, col_i + j, &tmp_ref, 0);
                    }
                }
            }
        }

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            for j in i..cols {
                if j != i {
                    // res_dft = (a[i] + a[j]) * (b[i] + b[j])
                    let (mut res_dft, mut scratch_3) = scratch.borrow().take_vec_znx_dft(self, 1, pairwise_dft_size);
                    {
                        let mut res_dft_backend = res_dft.reborrow_backend_mut();
                        self.cnv_pairwise_apply_dft(
                            cnv_offset_hi,
                            &mut res_dft_backend,
                            0,
                            &a_prep.reborrow_backend_ref(),
                            &b_prep.reborrow_backend_ref(),
                            i,
                            j,
                            &mut scratch_3,
                        );
                    }
                    let (mut res_big, scratch_4) = scratch_3.take_vec_znx_big(self, 1, pairwise_dft_size);
                    {
                        let mut res_big_backend = res_big.reborrow_backend_mut();
                        let mut res_dft_backend = res_dft.reborrow_backend_mut();
                        self.vec_znx_idft_apply_tmpa(&mut res_big_backend, 0, &mut res_dft_backend, 0);
                    }
                    let (mut tmp, mut scratch_5) = scratch_4.take_vec_znx(self.n(), 1, res.size());
                    let res_big_ref = res_big.reborrow_backend_ref();
                    let mut scratch_iter = scratch_5.borrow();
                    self.vec_znx_big_normalize(
                        &mut tmp,
                        res_base2k,
                        cnv_offset_lo,
                        0,
                        &res_big_ref,
                        ab_base2k,
                        0,
                        &mut scratch_iter,
                    );

                    let mut res_backend = <GLWETensor<R> as GLWEToBackendMut<BE>>::to_backend_mut(res);
                    let tmp_ref = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&tmp);
                    self.vec_znx_add_assign_backend(&mut res_backend.data, col_i + j, &tmp_ref, 0);
                }
            }
        }
    }

    fn glwe_tensor_apply_add_assign<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef,
    {
        assert!(
            scratch.available() >= self.glwe_tensor_apply_tmp_bytes(res, a, b),
            "scratch.available(): {} < GLWETensoring::glwe_tensor_apply_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_apply_tmp_bytes(res, a, b)
        );

        let ab_base2k: usize = a.base2k().as_usize();
        assert_eq!(b.base2k().as_usize(), ab_base2k);
        assert_eq!(a_effective_k.div_ceil(ab_base2k), a.size());
        assert_eq!(b_effective_k.div_ceil(ab_base2k), b.size());

        let res_base2k: usize = res.base2k().as_usize();
        let cols: usize = res.rank().as_usize() + 1;

        let (mut a_prep, scratch_1) = scratch.take_cnv_pvec_left(self, cols, a.size());
        let (mut b_prep, scratch_2) = scratch_1.take_cnv_pvec_right(self, cols, b.size());

        let a_mask = msb_mask_bottom_limb(ab_base2k, a_effective_k);
        let b_mask = msb_mask_bottom_limb(ab_base2k, b_effective_k);

        self.cnv_prepare_left(&mut a_prep, a.data(), a_mask, scratch_2);
        self.cnv_prepare_right(&mut b_prep, b.data(), b_mask, scratch_2);

        let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < ab_base2k {
            (0, -((ab_base2k - (cnv_offset % ab_base2k)) as i64))
        } else {
            ((cnv_offset / ab_base2k).saturating_sub(1), (cnv_offset % ab_base2k) as i64)
        };

        let diag_dft_size = normalize_input_limb_bound_with_offset(
            a.size() + b.size() - cnv_offset_hi,
            res.size(),
            res_base2k,
            ab_base2k,
            cnv_offset_lo,
        );
        let pairwise_dft_size = normalize_input_limb_bound_with_offset(
            a.size() + b.size() - cnv_offset_hi,
            res.size(),
            res_base2k,
            ab_base2k,
            cnv_offset_lo,
        );

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft(self, 1, diag_dft_size);
            self.cnv_apply_dft(cnv_offset_hi, &mut res_dft, 0, &a_prep, i, &b_prep, i, scratch_3);
            let res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);
            let (mut tmp, scratch_4) = scratch_3.take_vec_znx(self.n(), 1, res.size());
            self.vec_znx_big_normalize(&mut tmp, res_base2k, cnv_offset_lo, 0, &res_big, ab_base2k, 0, scratch_4);

            self.vec_znx_add_assign(res.data_mut(), col_i + i, &tmp, 0);

            for j in 0..cols {
                if j != i {
                    if j < i {
                        let col_j = j * cols - (j * (j + 1) / 2);
                        self.vec_znx_sub_assign(res.data_mut(), col_j + i, &tmp, 0);
                    } else {
                        self.vec_znx_sub_assign(res.data_mut(), col_i + j, &tmp, 0);
                    }
                }
            }
        }

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            for j in i..cols {
                if j != i {
                    let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft(self, 1, pairwise_dft_size);
                    self.cnv_pairwise_apply_dft(cnv_offset_hi, &mut res_dft, 0, &a_prep, &b_prep, i, j, scratch_3);
                    let res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);
                    let (mut tmp, scratch_3) = scratch_3.take_vec_znx(self.n(), 1, res.size());
                    self.vec_znx_big_normalize(&mut tmp, res_base2k, cnv_offset_lo, 0, &res_big, ab_base2k, 0, scratch_3);

                    self.vec_znx_add_assign(res.data_mut(), col_i + j, &tmp, 0);
                }
            }
        }
    }
}

#[inline]
pub fn msb_mask_bottom_limb(base2k: usize, k: usize) -> i64 {
    match k % base2k {
        0 => !0i64,
        r => (!0i64) << (base2k - r),
    }
}

#[inline]
fn normalize_input_limb_bound(
    full_size: usize,
    res_size: usize,
    res_base2k: usize,
    in_base2k: usize,
    offset_bits: usize,
) -> usize {
    full_size.min((res_size * res_base2k + offset_bits).div_ceil(in_base2k))
}

#[inline]
fn normalize_input_limb_bound_worst_case(full_size: usize, res_size: usize, res_base2k: usize, in_base2k: usize) -> usize {
    normalize_input_limb_bound(full_size, res_size, res_base2k, in_base2k, in_base2k - 1)
}

#[inline]
fn normalize_input_limb_bound_with_offset(
    full_size: usize,
    res_size: usize,
    res_base2k: usize,
    in_base2k: usize,
    res_offset: i64,
) -> usize {
    let mut offset_bits = res_offset % in_base2k as i64;
    if res_offset < 0 && offset_bits != 0 {
        offset_bits += in_base2k as i64;
    }
    normalize_input_limb_bound(full_size, res_size, res_base2k, in_base2k, offset_bits as usize)
}

impl<BE: Backend> GLWEAdd<BE> for Module<BE> where
    Self: ModuleN + VecZnxAddIntoBackend<BE> + VecZnxCopyBackend<BE> + VecZnxAddAssignBackend<BE> + VecZnxZeroBackend<BE>
{
}

impl<BE: Backend> GLWESub<BE> for Module<BE> where
    Self: ModuleN
        + VecZnxSubBackend<BE>
        + VecZnxSubAssignBackend<BE>
        + VecZnxSubNegateAssignBackend<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxNegateBackend<BE>
        + VecZnxZeroBackend<BE>
{
}

impl<BE: Backend> GLWENegate<BE> for Module<BE> where Self: VecZnxNegateBackend<BE> + VecZnxNegateAssignBackend<BE> + ModuleN {}

impl<BE: Backend> GLWERotateDefault<BE> for Module<BE> where
    Self:
        ModuleN + VecZnxRotateBackend<BE> + VecZnxRotateAssignBackend<BE> + VecZnxRotateAssignTmpBytes + VecZnxZeroBackend<BE>
{
}

#[doc(hidden)]
pub trait GLWERotateDefault<BE: Backend>
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
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_rotate_assign_backend(k, &mut res.data, i, &mut scratch_iter);
        }
    }
}

impl<BE: Backend> GLWEMulXpMinusOneDefault<BE> for Module<BE> where
    Self: ModuleN + VecZnxMulXpMinusOneBackend<BE> + VecZnxMulXpMinusOneAssignBackend<BE>
{
}

#[doc(hidden)]
pub trait GLWEMulXpMinusOneDefault<BE: Backend>
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
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_mul_xp_minus_one_assign_backend(k, &mut res.data, i, &mut scratch_iter);
        }
    }
}

impl<BE: Backend> GLWECopy<BE> for Module<BE> where Self: ModuleN + VecZnxCopyBackend<BE> + VecZnxZeroBackend<BE> {}

impl<BE: Backend> GLWEShiftDefault<BE> for Module<BE> where
    Self: ModuleN
        + VecZnxRshAssignBackend<BE>
        + VecZnxLshAddIntoBackend<BE>
        + VecZnxLshSubBackend<BE>
        + VecZnxRshTmpBytes
        + VecZnxLshTmpBytes
        + VecZnxLshAssignBackend<BE>
        + VecZnxLshBackend<BE>
{
}

#[doc(hidden)]
pub trait GLWEShiftDefault<BE: Backend>
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
        let lvl_0: usize = self.vec_znx_rsh_tmp_bytes().max(self.vec_znx_lsh_tmp_bytes());
        lvl_0
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
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_rsh_assign_backend(base2k, k, &mut res.data, i, &mut scratch_iter);
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
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_lsh_assign_backend(base2k, k, &mut res.data, i, &mut scratch_iter);
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
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_lsh_backend(base2k, k, &mut res.data, i, &a.data, i, &mut scratch_iter);
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
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_lsh_add_into_backend(base2k, k, &mut res.data, i, &a.data, i, &mut scratch_iter);
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
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_lsh_sub_backend(base2k, k, &mut res.data, i, &a.data, i, &mut scratch_iter);
        }
    }
}

impl<BE: Backend> GLWENormalizeDefault<BE> for Module<BE> where
    Self: ModuleN + VecZnxNormalize<BE> + VecZnxNormalizeAssignBackend<BE> + VecZnxNormalizeTmpBytes
{
}

#[doc(hidden)]
pub trait GLWENormalizeDefault<BE: Backend>
where
    Self: ModuleN + VecZnxNormalize<BE> + VecZnxNormalizeAssignBackend<BE> + VecZnxNormalizeTmpBytes,
{
    fn glwe_normalize_tmp_bytes(&self) -> usize {
        let lvl_0: usize = self.vec_znx_normalize_tmp_bytes();
        lvl_0
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
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_normalize(
                &mut res.data,
                res_base2k,
                0,
                i,
                &a.data,
                a.base2k().into(),
                i,
                &mut scratch_iter,
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
            let mut scratch_iter = scratch.borrow();
            self.vec_znx_normalize_assign_backend(res.base2k().into(), &mut res.data, i, &mut scratch_iter);
        }
    }
}
