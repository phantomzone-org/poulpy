use poulpy_hal::{
    api::{
        VecZnxAdd, VecZnxAddInplace, VecZnxCopy, VecZnxMulXpMinusOne, VecZnxMulXpMinusOneInplace, VecZnxNegateInplace,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxRotate, VecZnxRotateInplace, VecZnxRshInplace, VecZnxSub,
        VecZnxSubInplace, VecZnxSubNegateInplace,
    },
    layouts::{Backend, DataMut, Module, Scratch, VecZnx, ZnxZero},
};

use crate::layouts::{
    GLWECiphertext, GLWECiphertextToMut, GLWECiphertextToRef, GLWEInfos, GLWELayoutSet, GLWEPlaintext, LWEInfos, TorusPrecision,
};

impl<D> GLWEOperations for GLWEPlaintext<D>
where
    D: DataMut,
    GLWEPlaintext<D>: GLWECiphertextToMut + GLWEInfos,
{
}

impl<D: DataMut> GLWEOperations for GLWECiphertext<D> where GLWECiphertext<D>: GLWECiphertextToMut + GLWEInfos {}

pub trait GLWEOperations: GLWECiphertextToMut + GLWEInfos + GLWELayoutSet + Sized {
    fn add<A, B, BACKEND: Backend>(&mut self, module: &Module<BACKEND>, a: &A, b: &B)
    where
        A: GLWECiphertextToRef + GLWEInfos,
        B: GLWECiphertextToRef + GLWEInfos,
        Module<BACKEND>: VecZnxAdd + VecZnxCopy,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(a.base2k(), b.base2k());
            assert!(self.rank() >= a.rank().max(b.rank()));
        }

        let min_col: usize = (a.rank().min(b.rank()) + 1).into();
        let max_col: usize = (a.rank().max(b.rank() + 1)).into();
        let self_col: usize = (self.rank() + 1).into();

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();
        let b_ref: &GLWECiphertext<&[u8]> = &b.to_ref();

        (0..min_col).for_each(|i| {
            module.vec_znx_add(&mut self_mut.data, i, &a_ref.data, i, &b_ref.data, i);
        });

        if a.rank() > b.rank() {
            (min_col..max_col).for_each(|i| {
                module.vec_znx_copy(&mut self_mut.data, i, &a_ref.data, i);
            });
        } else {
            (min_col..max_col).for_each(|i| {
                module.vec_znx_copy(&mut self_mut.data, i, &b_ref.data, i);
            });
        }

        let size: usize = self_mut.size();
        (max_col..self_col).for_each(|i| {
            (0..size).for_each(|j| {
                self_mut.data.zero_at(i, j);
            });
        });

        self.set_basek(a.base2k());
        self.set_k(set_k_binary(self, a, b));
    }

    fn add_inplace<A, BACKEND: Backend>(&mut self, module: &Module<BACKEND>, a: &A)
    where
        A: GLWECiphertextToRef + GLWEInfos,
        Module<BACKEND>: VecZnxAddInplace,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(self.base2k(), a.base2k());
            assert!(self.rank() >= a.rank())
        }

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();

        (0..(a.rank() + 1).into()).for_each(|i| {
            module.vec_znx_add_inplace(&mut self_mut.data, i, &a_ref.data, i);
        });

        self.set_k(set_k_unary(self, a))
    }

    fn sub<A, B, BACKEND: Backend>(&mut self, module: &Module<BACKEND>, a: &A, b: &B)
    where
        A: GLWECiphertextToRef + GLWEInfos,
        B: GLWECiphertextToRef + GLWEInfos,
        Module<BACKEND>: VecZnxSub + VecZnxCopy + VecZnxNegateInplace,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(a.base2k(), b.base2k());
            assert!(self.rank() >= a.rank().max(b.rank()));
        }

        let min_col: usize = (a.rank().min(b.rank()) + 1).into();
        let max_col: usize = (a.rank().max(b.rank() + 1)).into();
        let self_col: usize = (self.rank() + 1).into();

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();
        let b_ref: &GLWECiphertext<&[u8]> = &b.to_ref();

        (0..min_col).for_each(|i| {
            module.vec_znx_sub(&mut self_mut.data, i, &a_ref.data, i, &b_ref.data, i);
        });

        if a.rank() > b.rank() {
            (min_col..max_col).for_each(|i| {
                module.vec_znx_copy(&mut self_mut.data, i, &a_ref.data, i);
            });
        } else {
            (min_col..max_col).for_each(|i| {
                module.vec_znx_copy(&mut self_mut.data, i, &b_ref.data, i);
                module.vec_znx_negate_inplace(&mut self_mut.data, i);
            });
        }

        let size: usize = self_mut.size();
        (max_col..self_col).for_each(|i| {
            (0..size).for_each(|j| {
                self_mut.data.zero_at(i, j);
            });
        });

        self.set_basek(a.base2k());
        self.set_k(set_k_binary(self, a, b));
    }

    fn sub_inplace_ab<A, BACKEND: Backend>(&mut self, module: &Module<BACKEND>, a: &A)
    where
        A: GLWECiphertextToRef + GLWEInfos,
        Module<BACKEND>: VecZnxSubInplace,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(self.base2k(), a.base2k());
            assert!(self.rank() >= a.rank())
        }

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();

        (0..(a.rank() + 1).into()).for_each(|i| {
            module.vec_znx_sub_inplace(&mut self_mut.data, i, &a_ref.data, i);
        });

        self.set_k(set_k_unary(self, a))
    }

    fn sub_inplace_ba<A, BACKEND: Backend>(&mut self, module: &Module<BACKEND>, a: &A)
    where
        A: GLWECiphertextToRef + GLWEInfos,
        Module<BACKEND>: VecZnxSubNegateInplace,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(self.base2k(), a.base2k());
            assert!(self.rank() >= a.rank())
        }

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();

        (0..(a.rank() + 1).into()).for_each(|i| {
            module.vec_znx_sub_negate_inplace(&mut self_mut.data, i, &a_ref.data, i);
        });

        self.set_k(set_k_unary(self, a))
    }

    fn rotate<A, B: Backend>(&mut self, module: &Module<B>, k: i64, a: &A)
    where
        A: GLWECiphertextToRef + GLWEInfos,
        Module<B>: VecZnxRotate,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(self.rank(), a.rank())
        }

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();

        (0..(a.rank() + 1).into()).for_each(|i| {
            module.vec_znx_rotate(k, &mut self_mut.data, i, &a_ref.data, i);
        });

        self.set_basek(a.base2k());
        self.set_k(set_k_unary(self, a))
    }

    fn rotate_inplace<B: Backend>(&mut self, module: &Module<B>, k: i64, scratch: &mut Scratch<B>)
    where
        Module<B>: VecZnxRotateInplace<B>,
    {
        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();

        (0..(self_mut.rank() + 1).into()).for_each(|i| {
            module.vec_znx_rotate_inplace(k, &mut self_mut.data, i, scratch);
        });
    }

    fn mul_xp_minus_one<A, B: Backend>(&mut self, module: &Module<B>, k: i64, a: &A)
    where
        A: GLWECiphertextToRef + GLWEInfos,
        Module<B>: VecZnxMulXpMinusOne,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(self.rank(), a.rank())
        }

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();

        (0..(a.rank() + 1).into()).for_each(|i| {
            module.vec_znx_mul_xp_minus_one(k, &mut self_mut.data, i, &a_ref.data, i);
        });

        self.set_basek(a.base2k());
        self.set_k(set_k_unary(self, a))
    }

    fn mul_xp_minus_one_inplace<B: Backend>(&mut self, module: &Module<B>, k: i64, scratch: &mut Scratch<B>)
    where
        Module<B>: VecZnxMulXpMinusOneInplace<B>,
    {
        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();

        (0..(self_mut.rank() + 1).into()).for_each(|i| {
            module.vec_znx_mul_xp_minus_one_inplace(k, &mut self_mut.data, i, scratch);
        });
    }

    fn copy<A, B: Backend>(&mut self, module: &Module<B>, a: &A)
    where
        A: GLWECiphertextToRef + GLWEInfos,
        Module<B>: VecZnxCopy,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), a.n());
            assert_eq!(self.rank(), a.rank());
        }

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();

        (0..(self_mut.rank() + 1).into()).for_each(|i| {
            module.vec_znx_copy(&mut self_mut.data, i, &a_ref.data, i);
        });

        self.set_k(a.k().min(self.max_k()));
        self.set_basek(a.base2k());
    }

    fn rsh<B: Backend>(&mut self, module: &Module<B>, k: usize, scratch: &mut Scratch<B>)
    where
        Module<B>: VecZnxRshInplace<B>,
    {
        let base2k: usize = self.base2k().into();
        (0..(self.rank() + 1).into()).for_each(|i| {
            module.vec_znx_rsh_inplace(base2k, k, &mut self.to_mut().data, i, scratch);
        })
    }

    fn normalize<A, B: Backend>(&mut self, module: &Module<B>, a: &A, scratch: &mut Scratch<B>)
    where
        A: GLWECiphertextToRef + GLWEInfos,
        Module<B>: VecZnxNormalize<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), a.n());
            assert_eq!(self.rank(), a.rank());
        }

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();

        (0..(self_mut.rank() + 1).into()).for_each(|i| {
            module.vec_znx_normalize(
                a.base2k().into(),
                &mut self_mut.data,
                i,
                a.base2k().into(),
                &a_ref.data,
                i,
                scratch,
            );
        });
        self.set_basek(a.base2k());
        self.set_k(a.k().min(self.k()));
    }

    fn normalize_inplace<B: Backend>(&mut self, module: &Module<B>, scratch: &mut Scratch<B>)
    where
        Module<B>: VecZnxNormalizeInplace<B>,
    {
        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        (0..(self_mut.rank() + 1).into()).for_each(|i| {
            module.vec_znx_normalize_inplace(self_mut.base2k().into(), &mut self_mut.data, i, scratch);
        });
    }
}

impl GLWECiphertext<Vec<u8>> {
    pub fn rsh_scratch_space(n: usize) -> usize {
        VecZnx::rsh_scratch_space(n)
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
