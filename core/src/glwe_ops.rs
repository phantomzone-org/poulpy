use backend::{Backend, Module, Scratch, VecZnx, VecZnxOps, VecZnxToMut, VecZnxToRef, ZnxZero};

use crate::elem::{Infos, SetMetaData};

pub trait GLWEOps<BACKEND: Backend>
where
    Self: Sized + VecZnxToMut + SetMetaData + Infos,
{
    fn add<A, B>(&mut self, module: &Module<BACKEND>, a: &A, b: &B)
    where
        A: VecZnxToRef + Infos,
        B: VecZnxToRef + Infos,
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

        (0..min_col).for_each(|i| {
            module.vec_znx_add(self, i, a, i, b, i);
        });

        if a.rank() > b.rank() {
            (min_col..max_col).for_each(|i| {
                module.vec_znx_copy(self, i, a, i);
            });
        } else {
            (min_col..max_col).for_each(|i| {
                module.vec_znx_copy(self, i, b, i);
            });
        }

        let size: usize = self.size();
        let mut self_mut: VecZnx<&mut [u8]> = self.to_mut();
        (max_col..self_col).for_each(|i| {
            (0..size).for_each(|j| {
                self_mut.zero_at(i, j);
            });
        });

        self.set_basek(a.basek());
        self.set_k(a.k().max(b.k()));
    }

    fn add_inplace<A>(&mut self, module: &Module<BACKEND>, a: &A)
    where
        A: VecZnxToRef + Infos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(self.basek(), a.basek());
            assert!(self.rank() >= a.rank())
        }

        (0..a.rank() + 1).for_each(|i| {
            module.vec_znx_add_inplace(self, i, a, i);
        });

        self.set_k(a.k().max(self.k()));
    }

    fn sub<A, B>(&mut self, module: &Module<BACKEND>, a: &A, b: &B)
    where
        A: VecZnxToRef + Infos,
        B: VecZnxToRef + Infos,
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

        (0..min_col).for_each(|i| {
            module.vec_znx_sub(self, i, a, i, b, i);
        });

        if a.rank() > b.rank() {
            (min_col..max_col).for_each(|i| {
                module.vec_znx_copy(self, i, a, i);
            });
        } else {
            (min_col..max_col).for_each(|i| {
                module.vec_znx_copy(self, i, b, i);
                module.vec_znx_negate_inplace(self, i);
            });
        }

        let size: usize = self.size();
        let mut self_mut: VecZnx<&mut [u8]> = self.to_mut();
        (max_col..self_col).for_each(|i| {
            (0..size).for_each(|j| {
                self_mut.zero_at(i, j);
            });
        });

        self.set_basek(a.basek());
        self.set_k(a.k().max(b.k()));
    }

    fn sub_inplace_ab<A>(&mut self, module: &Module<BACKEND>, a: &A)
    where
        A: VecZnxToRef + Infos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(self.basek(), a.basek());
            assert!(self.rank() >= a.rank())
        }

        (0..a.rank() + 1).for_each(|i| {
            module.vec_znx_sub_ab_inplace(self, i, a, i);
        });

        self.set_k(a.k().max(self.k()));
    }

    fn sub_inplace_ba<A>(&mut self, module: &Module<BACKEND>, a: &A)
    where
        A: VecZnxToRef + Infos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(self.basek(), a.basek());
            assert!(self.rank() >= a.rank())
        }

        (0..a.rank() + 1).for_each(|i| {
            module.vec_znx_sub_ba_inplace(self, i, a, i);
        });

        self.set_k(a.k().max(self.k()));
    }

    fn rotate<A>(&mut self, module: &Module<BACKEND>, k: i64, a: &A)
    where
        A: VecZnxToRef + Infos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(self.basek(), a.basek());
            assert_eq!(self.rank(), a.rank())
        }

        (0..a.rank() + 1).for_each(|i| {
            module.vec_znx_rotate(k, self, i, a, i);
        });

        self.set_k(a.k());
    }

    fn rotate_inplace<A>(&mut self, module: &Module<BACKEND>, k: i64)
    where
        A: VecZnxToRef + Infos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), module.n());
        }

        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_rotate_inplace(k, self, i);
        });
    }

    fn copy<A>(&mut self, module: &Module<BACKEND>, a: &A)
    where
        A: VecZnxToRef + Infos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), module.n());
            assert_eq!(a.n(), module.n());
        }

        let cols: usize = self.rank().min(a.rank()) + 1;

        (0..cols).for_each(|i| {
            module.vec_znx_copy(self, i, a, i);
        });

        self.set_k(a.k());
        self.set_basek(a.basek());
    }

    fn rsh(&mut self, k: usize, scratch: &mut Scratch) {
        let basek: usize = self.basek();
        let mut self_mut: VecZnx<&mut [u8]> = self.to_mut();
        self_mut.rsh(basek, k, scratch);
    }
}
