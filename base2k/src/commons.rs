use crate::{Backend, Module};

pub trait ZnxInfos {
    /// Returns the ring degree of the polynomials.
    fn n(&self) -> usize;

    /// Returns the base two logarithm of the ring dimension of the polynomials.
    fn log_n(&self) -> usize;

    /// Returns the number of rows.
    fn rows(&self) -> usize;

    /// Returns the number of polynomials in each row.
    fn cols(&self) -> usize;

    /// Returns the number of limbs per polynomial.
    fn limbs(&self) -> usize;

    /// Returns the total number of small polynomials.
    fn poly_count(&self) -> usize;
}

pub trait ZnxBase<B: Backend> {
    type Scalar;
    fn new(module: &Module<B>, cols: usize, limbs: usize) -> Self;
    fn from_bytes(module: &Module<B>, cols: usize, limbs: usize, bytes: &mut [u8]) -> Self;
    fn from_bytes_borrow(module: &Module<B>, cols: usize, limbs: usize, bytes: &mut [u8]) -> Self;
    fn bytes_of(module: &Module<B>, cols: usize, limbs: usize) -> usize;
}

pub trait ZnxLayout: ZnxInfos {
    type Scalar;

    /// Returns a non-mutable pointer to the underlying coefficients array.
    fn as_ptr(&self) -> *const Self::Scalar;

    /// Returns a mutable pointer to the underlying coefficients array.
    fn as_mut_ptr(&mut self) -> *mut Self::Scalar;

    /// Returns a non-mutable reference to the entire underlying coefficient array.
    fn raw(&self) -> &[Self::Scalar] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.n() * self.poly_count()) }
    }

    /// Returns a mutable reference to the entire underlying coefficient array.
    fn raw_mut(&mut self) -> &mut [Self::Scalar] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.n() * self.poly_count()) }
    }

    /// Returns a non-mutable pointer starting at the (i, j)-th small polynomial.
    fn at_ptr(&self, i: usize, j: usize) -> *const Self::Scalar {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.cols());
            assert!(j < self.limbs());
        }
        let offset = self.n() * (j * self.cols() + i);
        unsafe { self.as_ptr().add(offset) }
    }

    /// Returns a mutable pointer starting at the (i, j)-th small polynomial.
    fn at_mut_ptr(&mut self, i: usize, j: usize) -> *mut Self::Scalar {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.cols());
            assert!(j < self.limbs());
        }
        let offset = self.n() * (j * self.cols() + i);
        unsafe { self.as_mut_ptr().add(offset) }
    }

    /// Returns non-mutable reference to the (i, j)-th small polynomial.
    fn at_poly(&self, i: usize, j: usize) -> &[Self::Scalar] {
        unsafe { std::slice::from_raw_parts(self.at_ptr(i, j), self.n()) }
    }

    /// Returns mutable reference to the (i, j)-th small polynomial.
    fn at_poly_mut(&mut self, i: usize, j: usize) -> &mut [Self::Scalar] {
        unsafe { std::slice::from_raw_parts_mut(self.at_mut_ptr(i, j), self.n()) }
    }

    /// Returns non-mutable reference to the i-th limb.
    fn at_limb(&self, j: usize) -> &[Self::Scalar] {
        unsafe { std::slice::from_raw_parts(self.at_ptr(0, j), self.n() * self.cols()) }
    }

    /// Returns mutable reference to the i-th limb.
    fn at_limb_mut(&mut self, j: usize) -> &mut [Self::Scalar] {
        unsafe { std::slice::from_raw_parts_mut(self.at_mut_ptr(0, j), self.n() * self.cols()) }
    }
}
