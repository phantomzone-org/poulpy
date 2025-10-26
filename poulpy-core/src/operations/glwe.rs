use poulpy_hal::{
    api::{
        BivariateTensoring, ModuleN, ScratchTakeBasic, VecZnxAdd, VecZnxAddInplace, VecZnxBigNormalize, VecZnxCopy,
        VecZnxIdftApplyConsume, VecZnxMulXpMinusOne, VecZnxMulXpMinusOneInplace, VecZnxNegate, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxRotate, VecZnxRotateInplace, VecZnxRshInplace, VecZnxSub, VecZnxSubInplace,
        VecZnxSubNegateInplace, VecZnxZero,
    },
    layouts::{Backend, Module, Scratch, VecZnx, VecZnxBig, ZnxInfos},
    reference::vec_znx::vec_znx_rotate_inplace_tmp_bytes,
};

use crate::{
    ScratchTakeCore,
    layouts::{
        GLWE, GLWEInfos, GLWEPrepared, GLWEPreparedToRef, GLWETensor, GLWETensorToMut, GLWEToMut, GLWEToRef, LWEInfos,
        TorusPrecision,
    },
};

pub trait GLWETensoring<BE: Backend>
where
    Self: BivariateTensoring<BE> + VecZnxIdftApplyConsume<BE> + VecZnxBigNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    /// res = (a (x) b) * 2^{k * a_base2k}
    ///
    /// # Requires
    /// * a.base2k() == b.base2k()
    /// * res.cols() >= a.cols() + b.cols() - 1
    ///
    /// # Behavior
    /// * res precision is truncated to res.max_k().min(a.max_k() + b.max_k() + k * a_base2k)
    fn glwe_tensor<R, A, B>(&self, k: i64, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GLWETensorToMut,
        A: GLWEToRef,
        B: GLWEPreparedToRef<BE>,
    {
        let res: &mut GLWETensor<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();
        let b: &GLWEPrepared<&[u8], BE> = &b.to_ref();

        assert_eq!(a.base2k(), b.base2k());
        assert_eq!(a.rank(), res.rank());

        let res_cols: usize = res.data.cols();

        // Get tmp buffer of min precision between a_prec * b_prec and res_prec
        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self, res_cols, res.max_k().div_ceil(a.base2k()) as usize);

        // DFT(res) = DFT(a) (x) DFT(b)
        self.bivariate_tensoring(k, &mut res_dft, &a.data, &b.data, scratch_1);

        // res = IDFT(res)
        let res_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(res_dft);

        // Normalize and switches basis if required
        for res_col in 0..res_cols {
            self.vec_znx_big_normalize(
                res.base2k().into(),
                &mut res.data,
                res_col,
                a.base2k().into(),
                &res_big,
                res_col,
                scratch_1,
            );
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

impl<BE: Backend> GLWENormalize<BE> for Module<BE> where Self: ModuleN + VecZnxNormalize<BE> + VecZnxNormalizeInplace<BE> {}

pub trait GLWENormalize<BE: Backend>
where
    Self: ModuleN + VecZnxNormalize<BE> + VecZnxNormalizeInplace<BE>,
{
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

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_normalize(
                res.base2k().into(),
                res.data_mut(),
                i,
                a.base2k().into(),
                a.data(),
                i,
                scratch,
            );
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
