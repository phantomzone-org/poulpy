use poulpy_hal::{
    api::{
        ModuleN, VecZnxAdd, VecZnxAddInplace, VecZnxCopy, VecZnxMulXpMinusOne, VecZnxMulXpMinusOneInplace, VecZnxNegateInplace,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxRotate, VecZnxRotateInplace, VecZnxRshInplace, VecZnxSub,
        VecZnxSubInplace, VecZnxSubNegateInplace,
    },
    layouts::{Backend, Module, Scratch, VecZnx, ZnxZero},
};

use crate::{
    ScratchTakeCore,
    layouts::{GLWE, GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos, SetGLWEInfos, TorusPrecision},
};

pub trait GLWEAdd
where
    Self: ModuleN + VecZnxAdd + VecZnxCopy + VecZnxAddInplace,
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
        assert!(res.rank() >= a.rank().max(b.rank()));

        let min_col: usize = (a.rank().min(b.rank()) + 1).into();
        let max_col: usize = (a.rank().max(b.rank() + 1)).into();
        let self_col: usize = (res.rank() + 1).into();

        (0..min_col).for_each(|i| {
            self.vec_znx_add(res.data_mut(), i, a.data(), i, b.data(), i);
        });

        if a.rank() > b.rank() {
            (min_col..max_col).for_each(|i| {
                self.vec_znx_copy(res.data_mut(), i, a.data(), i);
            });
        } else {
            (min_col..max_col).for_each(|i| {
                self.vec_znx_copy(res.data_mut(), i, b.data(), i);
            });
        }

        let size: usize = res.size();
        (max_col..self_col).for_each(|i| {
            (0..size).for_each(|j| {
                res.data.zero_at(i, j);
            });
        });

        res.set_base2k(a.base2k());
        res.set_k(set_k_binary(res, a, b));
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

        (0..(a.rank() + 1).into()).for_each(|i| {
            self.vec_znx_add_inplace(res.data_mut(), i, a.data(), i);
        });

        res.set_k(set_k_unary(res, a))
    }
}

impl<BE: Backend> GLWEAdd for Module<BE> where Self: ModuleN + VecZnxAdd + VecZnxCopy + VecZnxAddInplace {}

pub trait GLWESub
where
    Self: ModuleN + VecZnxSub + VecZnxCopy + VecZnxNegateInplace + VecZnxSubInplace + VecZnxSubNegateInplace,
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
        assert_eq!(a.base2k(), b.base2k());
        assert!(res.rank() >= a.rank().max(b.rank()));

        let min_col: usize = (a.rank().min(b.rank()) + 1).into();
        let max_col: usize = (a.rank().max(b.rank() + 1)).into();
        let self_col: usize = (res.rank() + 1).into();

        (0..min_col).for_each(|i| {
            self.vec_znx_sub(res.data_mut(), i, a.data(), i, b.data(), i);
        });

        if a.rank() > b.rank() {
            (min_col..max_col).for_each(|i| {
                self.vec_znx_copy(res.data_mut(), i, a.data(), i);
            });
        } else {
            (min_col..max_col).for_each(|i| {
                self.vec_znx_copy(res.data_mut(), i, b.data(), i);
                self.vec_znx_negate_inplace(res.data_mut(), i);
            });
        }

        let size: usize = res.size();
        (max_col..self_col).for_each(|i| {
            (0..size).for_each(|j| {
                res.data.zero_at(i, j);
            });
        });

        res.set_base2k(a.base2k());
        res.set_k(set_k_binary(res, a, b));
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
        assert!(res.rank() >= a.rank());

        (0..(a.rank() + 1).into()).for_each(|i| {
            self.vec_znx_sub_inplace(res.data_mut(), i, a.data(), i);
        });

        res.set_k(set_k_unary(res, a))
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
        assert!(res.rank() >= a.rank());

        (0..(a.rank() + 1).into()).for_each(|i| {
            self.vec_znx_sub_negate_inplace(res.data_mut(), i, a.data(), i);
        });

        res.set_k(set_k_unary(res, a))
    }
}

pub trait GLWERotate<BE: Backend>
where
    Self: ModuleN + VecZnxRotate + VecZnxRotateInplace<BE>,
{
    fn glwe_rotate<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank());

        (0..(a.rank() + 1).into()).for_each(|i| {
            self.vec_znx_rotate(k, res.data_mut(), i, a.data(), i);
        });

        res.set_base2k(a.base2k());
        res.set_k(set_k_unary(res, a))
    }

    fn glwe_rotate_inplace<R>(&self, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        (0..(res.rank() + 1).into()).for_each(|i| {
            self.vec_znx_rotate_inplace(k, res.data_mut(), i, scratch);
        });
    }
}

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

        res.set_base2k(a.base2k());
        res.set_k(set_k_unary(res, a))
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

pub trait GLWECopy
where
    Self: ModuleN + VecZnxCopy,
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
        assert_eq!(res.rank(), a.rank());

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_copy(res.data_mut(), i, a.data(), i);
        }

        res.set_k(a.k().min(res.max_k()));
        res.set_base2k(a.base2k());
    }
}

pub trait GLWEShift<BE: Backend>
where
    Self: ModuleN + VecZnxRshInplace<BE>,
{
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
    pub fn rsh_tmp_bytes(n: usize) -> usize {
        VecZnx::rsh_tmp_bytes(n)
    }
}

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

        res.set_k(a.k().min(res.k()));
    }

    fn glwe_normalize_inplace<R>(&mut self, res: &mut R, scratch: &mut Scratch<BE>)
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

// a = op(a, b)
fn set_k_unary(a: &impl GLWEInfos, b: &impl GLWEInfos) -> TorusPrecision {
    if a.rank() != 0 || b.rank() != 0 {
        a.k().min(b.k())
    } else {
        a.k()
    }
}
