pub trait Infos {
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

pub trait VecZnxLayout: Infos {
    type Scalar;

    fn as_ptr(&self) -> *const Self::Scalar;
    fn as_mut_ptr(&mut self) -> *mut Self::Scalar;

    fn raw(&self) -> &[Self::Scalar] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.n() * self.poly_count()) }
    }

    fn raw_mut(&mut self) -> &mut [Self::Scalar] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.n() * self.poly_count()) }
    }

    fn at_ptr(&self, i: usize, j: usize) -> *const Self::Scalar {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.cols());
            assert!(j < self.limbs());
        }
        let offset = self.n() * (j * self.cols() + i);
        unsafe { self.as_ptr().add(offset) }
    }

    fn at_mut_ptr(&mut self, i: usize, j: usize) -> *mut Self::Scalar {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.cols());
            assert!(j < self.limbs());
        }
        let offset = self.n() * (j * self.cols() + i);
        unsafe { self.as_mut_ptr().add(offset) }
    }

    fn at_poly(&self, i: usize, j: usize) -> &[Self::Scalar] {
        unsafe { std::slice::from_raw_parts(self.at_ptr(i, j), self.n()) }
    }

    fn at_poly_mut(&mut self, i: usize, j: usize) -> &mut [Self::Scalar] {
        unsafe { std::slice::from_raw_parts_mut(self.at_mut_ptr(i, j), self.n()) }
    }

    fn at_limb(&self, j: usize) -> &[Self::Scalar] {
        unsafe { std::slice::from_raw_parts(self.at_ptr(0, j), self.n() * self.cols()) }
    }

    fn at_limb_mut(&mut self, j: usize) -> &mut [Self::Scalar] {
        unsafe { std::slice::from_raw_parts_mut(self.at_mut_ptr(0, j), self.n() * self.cols()) }
    }
}
