use backend::{FFT64, Module, Scratch, VecZnx, VecZnxOps, ZnxZero};

use crate::{
    elem::{Infos, SetMetaData},
    glwe_ciphertext::{GLWECiphertext, GLWECiphertextToMut, GLWECiphertextToRef},
};

pub trait GLWEOps: GLWECiphertextToMut + Infos + SetMetaData {
    fn add<A, B>(&mut self, module: &Module<FFT64>, a: &A, b: &B)
    where
        A: GLWECiphertextToRef + Infos,
        B: GLWECiphertextToRef + Infos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(b.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(a.basek(), b.basek());
            assert!(self.rank() >= a.rank().max(b.rank()));
        }

        let min_col: usize = a.rank().min(b.rank()) + 1;
        let max_col: usize = a.rank().max(b.rank() + 1);
        let self_col: usize = self.rank() + 1;

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

        self.set_basek(a.basek());
        self.set_k(a.k().max(b.k()));
    }

    fn add_inplace<A>(&mut self, module: &Module<FFT64>, a: &A)
    where
        A: GLWECiphertextToRef + Infos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(self.basek(), a.basek());
            assert!(self.rank() >= a.rank())
        }

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();

        (0..a.rank() + 1).for_each(|i| {
            module.vec_znx_add_inplace(&mut self_mut.data, i, &a_ref.data, i);
        });

        self.set_k(a.k().max(self.k()));
    }

    fn sub<A, B>(&mut self, module: &Module<FFT64>, a: &A, b: &B)
    where
        A: GLWECiphertextToRef + Infos,
        B: GLWECiphertextToRef + Infos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(b.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(a.basek(), b.basek());
            assert!(self.rank() >= a.rank().max(b.rank()));
        }

        let min_col: usize = a.rank().min(b.rank()) + 1;
        let max_col: usize = a.rank().max(b.rank() + 1);
        let self_col: usize = self.rank() + 1;

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

        self.set_basek(a.basek());
        self.set_k(a.k().max(b.k()));
    }

    fn sub_inplace_ab<A>(&mut self, module: &Module<FFT64>, a: &A)
    where
        A: GLWECiphertextToRef + Infos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(self.basek(), a.basek());
            assert!(self.rank() >= a.rank())
        }

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();

        (0..a.rank() + 1).for_each(|i| {
            module.vec_znx_sub_ab_inplace(&mut self_mut.data, i, &a_ref.data, i);
        });

        self.set_k(a.k().max(self.k()));
    }

    fn sub_inplace_ba<A>(&mut self, module: &Module<FFT64>, a: &A)
    where
        A: GLWECiphertextToRef + Infos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(self.basek(), a.basek());
            assert!(self.rank() >= a.rank())
        }

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();

        (0..a.rank() + 1).for_each(|i| {
            module.vec_znx_sub_ba_inplace(&mut self_mut.data, i, &a_ref.data, i);
        });

        self.set_k(a.k().max(self.k()));
    }

    fn rotate<A>(&mut self, module: &Module<FFT64>, k: i64, a: &A)
    where
        A: GLWECiphertextToRef + Infos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(self.rank(), a.rank())
        }

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();

        (0..a.rank() + 1).for_each(|i| {
            module.vec_znx_rotate(k, &mut self_mut.data, i, &a_ref.data, i);
        });

        self.set_basek(a.basek());
        self.set_k(a.k());
    }

    fn rotate_inplace(&mut self, module: &Module<FFT64>, k: i64) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), module.n());
        }

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();

        (0..self_mut.rank() + 1).for_each(|i| {
            module.vec_znx_rotate_inplace(k, &mut self_mut.data, i);
        });
    }

    fn copy<A>(&mut self, module: &Module<FFT64>, a: &A)
    where
        A: GLWECiphertextToRef + Infos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), module.n());
            assert_eq!(a.n(), module.n());
            assert_eq!(self.rank(), a.rank());
        }

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();

        (0..self_mut.rank() + 1).for_each(|i| {
            module.vec_znx_copy(&mut self_mut.data, i, &a_ref.data, i);
        });

        self.set_k(a.k());
        self.set_basek(a.basek());
    }

    fn rsh(&mut self, k: usize, scratch: &mut Scratch) {
        let basek: usize = self.basek();
        let mut self_mut: GLWECiphertext<&mut [u8]> = self.to_mut();
        self_mut.data.rsh(basek, k, scratch);
    }

    fn normalize<A>(&mut self, module: &Module<FFT64>, a: &A, scratch: &mut Scratch)
    where
        A: GLWECiphertextToRef + Infos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), module.n());
            assert_eq!(a.n(), module.n());
            assert_eq!(self.rank(), a.rank());
        }

        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        let a_ref: &GLWECiphertext<&[u8]> = &a.to_ref();

        (0..self_mut.rank() + 1).for_each(|i| {
            module.vec_znx_normalize(a.basek(), &mut self_mut.data, i, &a_ref.data, i, scratch);
        });
        self.set_basek(a.basek());
        self.set_k(a.k());
    }

    fn normalize_inplace(&mut self, module: &Module<FFT64>, scratch: &mut Scratch) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), module.n());
        }
        let self_mut: &mut GLWECiphertext<&mut [u8]> = &mut self.to_mut();
        (0..self_mut.rank() + 1).for_each(|i| {
            module.vec_znx_normalize_inplace(self_mut.basek(), &mut self_mut.data, i, scratch);
        });
    }
}

impl GLWECiphertext<Vec<u8>> {
    pub fn rsh_scratch_space(module: &Module<FFT64>) -> usize {
        VecZnx::rsh_scratch_space(module.n())
    }
}
