use itertools::izip;
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

pub trait ZnxSliceSize {
    /// Returns the slice size, which is the offset between
    /// two size of the same column.
    fn sl(&self) -> usize;
}

pub trait DataView {
    type D;
    fn data(&self) -> &Self::D;
}

pub trait DataViewMut: DataView {
    fn data_mut(&mut self) -> &mut Self::D;
}

pub trait ZnxView: ZnxInfos + DataView<D: AsRef<[u8]>> {
    type Scalar: Copy;

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
            assert!(i < self.cols());
            assert!(j < self.size());
        }
        let offset: usize = self.n() * (j * self.cols() + i);
        unsafe { self.as_ptr().add(offset) }
    }

    /// Returns non-mutable reference to the (i, j)-th small polynomial.
    fn at(&self, i: usize, j: usize) -> &[Self::Scalar] {
        unsafe { std::slice::from_raw_parts(self.at_ptr(i, j), self.n()) }
    }
}

pub trait ZnxViewMut: ZnxView + DataViewMut<D: AsMut<[u8]>> {
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
            assert!(i < self.cols());
            assert!(j < self.size());
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
impl<T> ZnxViewMut for T where T: ZnxView + DataViewMut<D: AsMut<[u8]>> {}

pub trait ZnxZero: ZnxViewMut
where
    Self: Sized,
{
    fn zero(&mut self) {
        unsafe {
            std::ptr::write_bytes(
                self.as_mut_ptr(),
                0,
                self.n() * self.poly_count(),
            );
        }
    }

    fn zero_at(&mut self, i: usize, j: usize) {
        unsafe {
            std::ptr::write_bytes(
                self.at_mut_ptr(i, j),
                0,
                self.n(),
            );
        }
    }
}

// Blanket implementations
impl<T> ZnxZero for T where T: ZnxViewMut {}

use std::ops::{Add, AddAssign, Div, Mul, Neg, Shl, Shr, Sub};

use crate::Scratch;
pub trait Integer:
    Copy
    + Default
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Shl<Output = Self>
    + Shr<Output = Self>
    + AddAssign
{
    const BITS: u32;
}

impl Integer for i64 {
    const BITS: u32 = 64;
}

impl Integer for i128 {
    const BITS: u32 = 128;
}

//(Jay)Note: `rsh` impl. ignores the column
pub fn rsh<V: ZnxZero>(k: usize, log_base2k: usize, a: &mut V, _a_col: usize, scratch: &mut Scratch)
where
    V::Scalar: From<usize> + Integer + Zero,
{
    let n: usize = a.n();
    let _size: usize = a.size();
    let cols: usize = a.cols();

    let size: usize = a.size();
    let steps: usize = k / log_base2k;

    a.raw_mut().rotate_right(n * steps * cols);
    (0..cols).for_each(|i| {
        (0..steps).for_each(|j| {
            a.zero_at(i, j);
        })
    });

    let k_rem: usize = k % log_base2k;

    if k_rem != 0 {
        let (carry, _) = scratch.tmp_scalar_slice::<V::Scalar>(rsh_tmp_bytes::<V::Scalar>(n));

        unsafe {
            std::ptr::write_bytes(carry.as_mut_ptr(), 0, n * size_of::<V::Scalar>());
        }

        let log_base2k_t = V::Scalar::from(log_base2k);
        let shift = V::Scalar::from(V::Scalar::BITS as usize - k_rem);
        let k_rem_t = V::Scalar::from(k_rem);

        (0..cols).for_each(|i| {
            (steps..size).for_each(|j| {
                izip!(carry.iter_mut(), a.at_mut(i, j).iter_mut()).for_each(|(ci, xi)| {
                    *xi += *ci << log_base2k_t;
                    *ci = (*xi << shift) >> shift;
                    *xi = (*xi - *ci) >> k_rem_t;
                });
            });
            carry.iter_mut().for_each(|r| *r = V::Scalar::zero());
        })
    }
}

pub fn rsh_tmp_bytes<T>(n: usize) -> usize {
    n * std::mem::size_of::<T>()
}
