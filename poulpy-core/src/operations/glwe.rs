use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, ScratchTakeBasic, VecZnxAdd, VecZnxAddInplace, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxMulXpMinusOne,
        VecZnxMulXpMinusOneInplace, VecZnxNegate, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate,
        VecZnxRotateInplace, VecZnxRshInplace, VecZnxSub, VecZnxSubInplace, VecZnxSubNegateInplace, VecZnxZero,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, VecZnx, VecZnxBig},
    reference::vec_znx::vec_znx_rotate_inplace_tmp_bytes,
};

use crate::{
    ScratchTakeCore,
    layouts::{GLWE, GLWEInfos, GLWETensor, GLWEToMut, GLWEToRef, LWEInfos, TorusPrecision},
};

pub trait GLWETensoring<BE: Backend> {
    fn glwe_tensor_tmp_bytes<R, A, B>(&self, res: &R, res_offset: usize, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    /// res = (a (x) b) * 2^{res_offset * a_base2k}
    ///
    /// # Requires
    /// * a.base2k() == b.base2k()
    /// * a.rank() == b.rank()
    ///
    /// # Behavior
    /// * res precision is truncated to res.max_k().min(a.max_k() + b.max_k() + k * a_base2k)
    fn glwe_tensor<R, A, B>(
        &self,
        res: &mut GLWETensor<R>,
        res_offset: usize,
        a: &GLWE<A>,
        b: &GLWE<B>,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;
}

impl<BE: Backend> GLWETensoring<BE> for Module<BE>
where
    Self: Sized
        + ModuleN
        + CnvPVecBytesOf
        + VecZnxDftBytesOf
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + Convolution<BE>
        + VecZnxSubInplace
        + VecZnxNegate
        + VecZnxAddInplace
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopy,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_tensor_tmp_bytes<R, A, B>(&self, res: &R, res_offset: usize, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        let cols: usize = res.rank().as_usize() + 1;

        let a_size: usize = a.size();
        let b_size: usize = b.size();
        let res_size: usize = res.size();

        let cnv_pvec: usize = self.bytes_of_cnv_pvec_left(cols, a_size) + self.bytes_of_cnv_pvec_right(cols, b_size);
        let cnv_prep: usize = self
            .cnv_prepare_left_tmp_bytes(a_size, a_size)
            .max(self.cnv_prepare_left_tmp_bytes(a_size, a_size));
        let cnv_apply: usize = self
            .cnv_apply_dft_tmp_bytes(res_size, res_offset, a_size, b_size)
            .max(self.cnv_pairwise_apply_dft_tmp_bytes(res_size, res_offset, a_size, b_size));

        let res_dft_size = res
            .k()
            .as_usize()
            .div_ceil(a.base2k().as_usize())
            .min(a_size + b_size - res_offset);

        let res_dft: usize = self.bytes_of_vec_znx_dft(1, res_dft_size);
        let tmp: usize = VecZnx::bytes_of(self.n(), 1, res.size());
        let norm: usize = self.vec_znx_big_normalize_tmp_bytes();

        cnv_pvec + cnv_prep + res_dft + cnv_apply.max(tmp + norm)
    }

    fn glwe_tensor<R, A, B>(
        &self,
        res: &mut GLWETensor<R>,
        res_offset: usize,
        a: &GLWE<A>,
        b: &GLWE<B>,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef,
    {
        let a_base2k: usize = a.base2k().as_usize();
        assert_eq!(b.base2k().as_usize(), a_base2k);
        let res_base2k: usize = res.base2k().as_usize();

        let cols: usize = res.rank().as_usize() + 1;

        let (mut a_prep, scratch_1) = scratch.take_cnv_pvec_left(self, cols, a.size());
        let (mut b_prep, scratch_2) = scratch_1.take_cnv_pvec_right(self, cols, b.size());

        self.cnv_prepare_left(&mut a_prep, a.data(), scratch_2);
        self.cnv_prepare_right(&mut b_prep, b.data(), scratch_2);

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

        // Ceil the lsh offset during conv, because we then correct
        // with a rsh.
        let res_offset_conv: usize = res_offset.div_ceil(a_base2k);
        let res_offset_rsh: usize = (a_base2k - res_offset % a_base2k) % a_base2k;

        let res_dft_size = res
            .k()
            .as_usize()
            .div_ceil(a.base2k().as_usize())
            .min(a.size() + b.size() - res_offset_conv);

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft(self, 1, res_dft_size);
            self.cnv_apply_dft(&mut res_dft, res_offset_conv, 0, &a_prep, i, &b_prep, i, scratch_3);
            let res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);
            let (mut tmp, scratch_4) = scratch_3.take_vec_znx(self.n(), 1, res_dft_size);
            self.vec_znx_big_normalize(
                &mut tmp,
                res_base2k,
                -(res_offset_rsh as i64),
                0,
                &res_big,
                a_base2k,
                0,
                scratch_4,
            );

            self.vec_znx_copy(res.data_mut(), col_i + i, &tmp, 0);

            // Pre-subtracts
            // res[i!=j] = NEG(a[i] * b[i]) + SUB(a[j] * b[j])
            for j in i + 1..cols {
                if col_i < j {
                    self.vec_znx_sub_inplace(res.data_mut(), col_i + j, &tmp, 0);
                } else {
                    self.vec_znx_negate(res.data_mut(), col_i + j, &tmp, 0);
                }
            }
        }

        for i in 0..cols {
            let col_i: usize = i * cols - (i * (i + 1) / 2);

            for j in i + 1..cols {
                // res_dft = (a[i] + a[j]) * (b[i] + b[j])
                let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft(self, 1, res.size());
                self.cnv_pairwise_apply_dft(&mut res_dft, res_offset, 0, &a_prep, &b_prep, i, j, scratch_3);
                let res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);
                let (mut tmp, scratch_3) = scratch_3.take_vec_znx(self.n(), 1, res.size());
                self.vec_znx_big_normalize(
                    &mut tmp,
                    res_base2k,
                    -(res_offset_rsh as i64),
                    0,
                    &res_big,
                    a_base2k,
                    0,
                    scratch_3,
                );
                self.vec_znx_add_inplace(res.data_mut(), col_i + j, &tmp, 0);
            }
        }
    }
}

pub trait GLWEAdd
where
    Self: ModuleN + VecZnxAdd + VecZnxCopy + VecZnxAddInplace + VecZnxZero,
{
    fn glwe_add<R, A, B>(&self, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        B: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &mut GLWE<&[u8]> = &mut a.to_ref();
        let b: &GLWE<&[u8]> = &b.to_ref();

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
        let max_col: usize = (a.rank().max(b.rank() + 1)).into();
        let self_col: usize = (res.rank() + 1).into();

        for i in 0..min_col {
            self.vec_znx_add(res.data_mut(), i, a.data(), i, b.data(), i);
        }

        if a.rank() > b.rank() {
            for i in min_col..max_col {
                self.vec_znx_copy(res.data_mut(), i, a.data(), i);
            }
        } else {
            for i in min_col..max_col {
                self.vec_znx_copy(res.data_mut(), i, b.data(), i);
            }
        }

        for i in max_col..self_col {
            self.vec_znx_zero(res.data_mut(), i);
        }
    }

    fn glwe_add_inplace<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() >= a.rank());

        for i in 0..(a.rank() + 1).into() {
            self.vec_znx_add_inplace(res.data_mut(), i, a.data(), i);
        }
    }
}

impl<BE: Backend> GLWEAdd for Module<BE> where Self: ModuleN + VecZnxAdd + VecZnxCopy + VecZnxAddInplace + VecZnxZero {}

impl<BE: Backend> GLWESub for Module<BE> where
    Self: ModuleN + VecZnxSub + VecZnxCopy + VecZnxNegate + VecZnxZero + VecZnxSubInplace + VecZnxSubNegateInplace
{
}

pub trait GLWESub
where
    Self: ModuleN + VecZnxSub + VecZnxCopy + VecZnxNegate + VecZnxZero + VecZnxSubInplace + VecZnxSubNegateInplace,
{
    fn glwe_sub<R, A, B>(&self, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        B: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();
        let b: &GLWE<&[u8]> = &b.to_ref();

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
        let max_col: usize = (a.rank().max(b.rank() + 1)).into();
        let self_col: usize = (res.rank() + 1).into();

        for i in 0..min_col {
            self.vec_znx_sub(res.data_mut(), i, a.data(), i, b.data(), i);
        }

        if a.rank() > b.rank() {
            for i in min_col..max_col {
                self.vec_znx_copy(res.data_mut(), i, a.data(), i);
            }
        } else {
            for i in min_col..max_col {
                self.vec_znx_negate(res.data_mut(), i, b.data(), i);
            }
        }

        for i in max_col..self_col {
            self.vec_znx_zero(res.data_mut(), i);
        }
    }

    fn glwe_sub_inplace<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() == a.rank() || a.rank() == 0);

        for i in 0..(a.rank() + 1).into() {
            self.vec_znx_sub_inplace(res.data_mut(), i, a.data(), i);
        }
    }

    fn glwe_sub_negate_inplace<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() == a.rank() || a.rank() == 0);

        for i in 0..(a.rank() + 1).into() {
            self.vec_znx_sub_negate_inplace(res.data_mut(), i, a.data(), i);
        }
    }
}

impl<BE: Backend> GLWERotate<BE> for Module<BE> where Self: ModuleN + VecZnxRotate + VecZnxRotateInplace<BE> + VecZnxZero {}

pub trait GLWERotate<BE: Backend>
where
    Self: ModuleN + VecZnxRotate + VecZnxRotateInplace<BE> + VecZnxZero,
{
    fn glwe_rotate_tmp_bytes(&self) -> usize {
        vec_znx_rotate_inplace_tmp_bytes(self.n())
    }

    fn glwe_rotate<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.n(), self.n() as u32);
        assert!(res.rank() == a.rank() || a.rank() == 0);

        let res_cols = (res.rank() + 1).into();
        let a_cols = (a.rank() + 1).into();

        for i in 0..a_cols {
            self.vec_znx_rotate(k, res.data_mut(), i, a.data(), i);
        }
        for i in a_cols..res_cols {
            self.vec_znx_zero(res.data_mut(), i);
        }
    }

    fn glwe_rotate_inplace<R>(&self, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        for i in 0..(res.rank() + 1).into() {
            self.vec_znx_rotate_inplace(k, res.data_mut(), i, scratch);
        }
    }
}

impl<BE: Backend> GLWEMulXpMinusOne<BE> for Module<BE> where Self: ModuleN + VecZnxMulXpMinusOne + VecZnxMulXpMinusOneInplace<BE> {}

pub trait GLWEMulXpMinusOne<BE: Backend>
where
    Self: ModuleN + VecZnxMulXpMinusOne + VecZnxMulXpMinusOneInplace<BE>,
{
    fn glwe_mul_xp_minus_one<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank());

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_mul_xp_minus_one(k, res.data_mut(), i, a.data(), i);
        }
    }

    fn glwe_mul_xp_minus_one_inplace<R>(&self, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        assert_eq!(res.n(), self.n() as u32);

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_mul_xp_minus_one_inplace(k, res.data_mut(), i, scratch);
        }
    }
}

impl<BE: Backend> GLWECopy for Module<BE> where Self: ModuleN + VecZnxCopy + VecZnxZero {}

pub trait GLWECopy
where
    Self: ModuleN + VecZnxCopy + VecZnxZero,
{
    fn glwe_copy<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert!(res.rank() == a.rank() || a.rank() == 0);

        let min_rank: usize = res.rank().min(a.rank()).as_usize() + 1;

        for i in 0..min_rank {
            self.vec_znx_copy(res.data_mut(), i, a.data(), i);
        }

        for i in min_rank..(res.rank() + 1).into() {
            self.vec_znx_zero(res.data_mut(), i);
        }
    }
}

impl<BE: Backend> GLWEShift<BE> for Module<BE> where Self: ModuleN + VecZnxRshInplace<BE> {}

pub trait GLWEShift<BE: Backend>
where
    Self: ModuleN + VecZnxRshInplace<BE>,
{
    fn glwe_rsh_tmp_byte(&self) -> usize {
        VecZnx::rsh_tmp_bytes(self.n())
    }

    fn glwe_rsh<R>(&self, k: usize, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let base2k: usize = res.base2k().into();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_rsh_inplace(base2k, k, res.data_mut(), i, scratch);
        }
    }
}

impl GLWE<Vec<u8>> {
    pub fn rsh_tmp_bytes<M, BE: Backend>(module: &M) -> usize
    where
        M: GLWEShift<BE>,
    {
        module.glwe_rsh_tmp_byte()
    }
}

impl<BE: Backend> GLWENormalize<BE> for Module<BE> where
    Self: ModuleN + VecZnxNormalize<BE> + VecZnxNormalizeInplace<BE> + VecZnxNormalizeTmpBytes
{
}

pub trait GLWENormalize<BE: Backend>
where
    Self: ModuleN + VecZnxNormalize<BE> + VecZnxNormalizeInplace<BE> + VecZnxNormalizeTmpBytes,
{
    fn glwe_normalize_tmp_bytes(&self) -> usize {
        self.vec_znx_normalize_tmp_bytes()
    }

    /// Usage:
    /// let mut tmp_b: Option<GLWE<&mut [u8]>> = None;
    /// let (b_conv, scratch_1) = glwe_maybe_convert_in_place(self, b, res.base2k().as_u32(), &mut tmp_b, scratch);
    fn glwe_maybe_cross_normalize_to_ref<'a, A>(
        &self,
        glwe: &'a A,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWE<&'a mut [u8]>>, // caller-owned scratch-backed temp
        scratch: &'a mut Scratch<BE>,
    ) -> (GLWE<&'a [u8]>, &'a mut Scratch<BE>)
    where
        A: GLWEToRef + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        // No conversion: just use the original GLWE
        if glwe.base2k().as_usize() == target_base2k {
            // Drop any previous temp; it's stale for this base
            tmp_slot.take();
            return (glwe.to_ref(), scratch);
        }

        // Conversion: allocate a temporary GLWE in scratch
        let mut layout = glwe.glwe_layout();
        layout.base2k = target_base2k.into();

        let (tmp, scratch2) = scratch.take_glwe(&layout);
        *tmp_slot = Some(tmp);

        // Get a mutable handle to the temp and normalize into it
        let tmp_ref: &mut GLWE<&mut [u8]> = tmp_slot.as_mut().expect("tmp_slot just set to Some, but found None");

        self.glwe_normalize(tmp_ref, glwe, scratch2);

        // Return a trait-object view of the temp
        (tmp_ref.to_ref(), scratch2)
    }

    /// Usage:
    /// let mut tmp_b: Option<GLWE<&mut [u8]>> = None;
    /// let (b_conv, scratch_1) = glwe_maybe_convert_in_place(self, b, res.base2k().as_u32(), &mut tmp_b, scratch);
    fn glwe_maybe_cross_normalize_to_mut<'a, A>(
        &self,
        glwe: &'a mut A,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWE<&'a mut [u8]>>, // caller-owned scratch-backed temp
        scratch: &'a mut Scratch<BE>,
    ) -> (GLWE<&'a mut [u8]>, &'a mut Scratch<BE>)
    where
        A: GLWEToMut + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        // No conversion: just use the original GLWE
        if glwe.base2k().as_usize() == target_base2k {
            // Drop any previous temp; it's stale for this base
            tmp_slot.take();
            return (glwe.to_mut(), scratch);
        }

        // Conversion: allocate a temporary GLWE in scratch
        let mut layout = glwe.glwe_layout();
        layout.base2k = target_base2k.into();

        let (tmp, scratch2) = scratch.take_glwe(&layout);
        *tmp_slot = Some(tmp);

        // Get a mutable handle to the temp and normalize into it
        let tmp_ref: &mut GLWE<&mut [u8]> = tmp_slot.as_mut().expect("tmp_slot just set to Some, but found None");

        self.glwe_normalize(tmp_ref, glwe, scratch2);

        // Return a trait-object view of the temp
        (tmp_ref.to_mut(), scratch2)
    }

    fn glwe_normalize<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank());

        let res_base2k = res.base2k().into();

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_normalize(res.data_mut(), res_base2k, 0, i, a.data(), a.base2k().into(), i, scratch);
        }
    }

    fn glwe_normalize_inplace<R>(&self, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_normalize_inplace(res.base2k().into(), res.data_mut(), i, scratch);
        }
    }
}

#[allow(dead_code)]
// c = op(a, b)
fn set_k_binary(c: &impl GLWEInfos, a: &impl GLWEInfos, b: &impl GLWEInfos) -> TorusPrecision {
    // If either operands is a ciphertext
    if a.rank() != 0 || b.rank() != 0 {
        // If a is a plaintext (but b ciphertext)
        let k = if a.rank() == 0 {
            b.k()
        // If b is a plaintext (but a ciphertext)
        } else if b.rank() == 0 {
            a.k()
        // If a & b are both ciphertexts
        } else {
            a.k().min(b.k())
        };
        k.min(c.k())
    // If a & b are both plaintexts
    } else {
        c.k()
    }
}

#[allow(dead_code)]
// a = op(a, b)
fn set_k_unary(a: &impl GLWEInfos, b: &impl GLWEInfos) -> TorusPrecision {
    if a.rank() != 0 || b.rank() != 0 {
        a.k().min(b.k())
    } else {
        a.k()
    }
}
