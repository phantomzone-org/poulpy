use crate::{
    layouts::{Backend, Data, DataMut, DataRef},
    source::Source,
};
use rand_distr::num_traits::Zero;

pub trait ZnxInfos {
    /// Returns the ring degree of the polynomials.
    fn n(&self) -> usize;

    /// Returns the base two logarithm of the ring dimension of the polynomials.
    fn log_n(&self) -> usize {
        (usize::BITS - (self.n() - 1).leading_zeros()) as _
    }

    /// Returns the number of rows.
    fn rows(&self) -> usize;

    /// Returns the number of polynomials in each row.
    fn cols(&self) -> usize;

    /// Returns the number of size per polynomial.
    fn size(&self) -> usize;

    /// Returns the total number of small polynomials.
    fn poly_count(&self) -> usize {
        self.rows() * self.cols() * self.size()
    }
}

pub trait ZnxSliceSizeImpl<B: Backend> {
    fn slice_size(&self) -> usize;
}

pub trait ZnxSliceSize {
    /// Returns the slice size, which is the offset between
    /// two size of the same column.
    fn sl(&self) -> usize;
}

pub trait DataView {
    type D: Data;
    fn data(&self) -> &Self::D;
}

pub trait DataViewMut: DataView {
    fn data_mut(&mut self) -> &mut Self::D;
}

pub trait ZnxView: ZnxInfos + DataView<D: DataRef> {
    type Scalar: Copy + Zero;

    /// Returns a non-mutable pointer to the underlying coefficients array.
    fn as_ptr(&self) -> *const Self::Scalar {
        self.data().as_ref().as_ptr() as *const Self::Scalar
    }

    /// Returns a non-mutable reference to the entire underlying coefficient array.
    fn raw(&self) -> &[Self::Scalar] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.n() * self.poly_count()) }
    }

    /// Returns a non-mutable pointer starting at the j-th small polynomial of the i-th column.
    fn at_ptr(&self, i: usize, j: usize) -> *const Self::Scalar {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.cols(), "cols: {} >= {}", i, self.cols());
            assert!(j < self.size(), "size: {} >= {}", j, self.size());
        }
        let offset: usize = self.n() * (j * self.cols() + i);
        unsafe { self.as_ptr().add(offset) }
    }

    /// Returns non-mutable reference to the (i, j)-th small polynomial.
    fn at(&self, i: usize, j: usize) -> &[Self::Scalar] {
        unsafe { std::slice::from_raw_parts(self.at_ptr(i, j), self.n()) }
    }
}

pub trait ZnxViewMut: ZnxView + DataViewMut<D: DataMut> {
    /// Returns a mutable pointer to the underlying coefficients array.
    fn as_mut_ptr(&mut self) -> *mut Self::Scalar {
        self.data_mut().as_mut().as_mut_ptr() as *mut Self::Scalar
    }

    /// Returns a mutable reference to the entire underlying coefficient array.
    fn raw_mut(&mut self) -> &mut [Self::Scalar] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.n() * self.poly_count()) }
    }

    /// Returns a mutable pointer starting at the j-th small polynomial of the i-th column.
    fn at_mut_ptr(&mut self, i: usize, j: usize) -> *mut Self::Scalar {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.cols(), "cols: {} >= {}", i, self.cols());
            assert!(j < self.size(), "size: {} >= {}", j, self.size());
        }
        let offset: usize = self.n() * (j * self.cols() + i);
        unsafe { self.as_mut_ptr().add(offset) }
    }

    /// Returns mutable reference to the (i, j)-th small polynomial.
    fn at_mut(&mut self, i: usize, j: usize) -> &mut [Self::Scalar] {
        unsafe { std::slice::from_raw_parts_mut(self.at_mut_ptr(i, j), self.n()) }
    }
}

//(Jay)Note: Can't provide blanket impl. of ZnxView because Scalar is not known
impl<T> ZnxViewMut for T where T: ZnxView + DataViewMut<D: DataMut> {}

pub trait ZnxZero
where
    Self: Sized,
{
    fn zero(&mut self);
    fn zero_at(&mut self, i: usize, j: usize);
}

pub trait FillUniform {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source);
}
