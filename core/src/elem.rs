use base2k::{Backend, Module, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, ZnxInfos};

use crate::{glwe::GLWECiphertextFourier, utils::derive_size};

pub trait Infos {
    type Inner: ZnxInfos;

    fn inner(&self) -> &Self::Inner;

    /// Returns the ring degree of the polynomials.
    fn n(&self) -> usize {
        self.inner().n()
    }

    /// Returns the base two logarithm of the ring dimension of the polynomials.
    fn log_n(&self) -> usize {
        self.inner().log_n()
    }

    /// Returns the number of rows.
    fn rows(&self) -> usize {
        self.inner().rows()
    }

    /// Returns the number of polynomials in each row.
    fn rank(&self) -> usize {
        self.inner().cols()
    }

    /// Returns the number of size per polynomial.
    fn size(&self) -> usize {
        let size: usize = self.inner().size();
        debug_assert_eq!(size, derive_size(self.basek(), self.k()));
        size
    }

    /// Returns the total number of small polynomials.
    fn poly_count(&self) -> usize {
        self.rows() * self.rank() * self.size()
    }

    /// Returns the base 2 logarithm of the ciphertext base.
    fn basek(&self) -> usize;

    /// Returns the bit precision of the ciphertext.
    fn k(&self) -> usize;
}

pub trait GetRow<B: Backend> {
    fn get_row<R>(&self, module: &Module<B>, row_i: usize, col_j: usize, res: &mut GLWECiphertextFourier<R, B>)
    where
        VecZnxDft<R, B>: VecZnxDftToMut<B>;
}

pub trait SetRow<B: Backend> {
    fn set_row<A>(&mut self, module: &Module<B>, row_i: usize, col_j: usize, a: &GLWECiphertextFourier<A, B>)
    where
        VecZnxDft<A, B>: VecZnxDftToRef<B>;
}
