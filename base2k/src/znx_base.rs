use crate::{Backend, Module, alloc_aligned, assert_alignement, cast_mut};
use itertools::izip;
use std::cmp::min;

pub struct ZnxBase {
    /// The ring degree
    pub n: usize,

    /// The number of rows (in the third dimension)
    pub rows: usize,

    /// The number of polynomials
    pub cols: usize,

    /// The number of size per polynomial (a.k.a small polynomials).
    pub size: usize,

    /// Polynomial coefficients, as a contiguous array. Each col is equally spaced by n.
    pub data: Vec<u8>,

    /// Pointer to data (data can be enpty if [VecZnx] borrows space instead of owning it).
    pub ptr: *mut u8,
}

pub trait GetZnxBase {
    fn znx(&self) -> &ZnxBase;
    fn znx_mut(&mut self) -> &mut ZnxBase;
}

pub trait ZnxInfos: GetZnxBase {
    /// Returns the ring degree of the polynomials.
    fn n(&self) -> usize {
        self.znx().n
    }

    /// Returns the base two logarithm of the ring dimension of the polynomials.
    fn log_n(&self) -> usize {
        (usize::BITS - (self.n() - 1).leading_zeros()) as _
    }

    /// Returns the number of rows.
    fn rows(&self) -> usize {
        self.znx().rows
    }
    /// Returns the number of polynomials in each row.
    fn cols(&self) -> usize {
        self.znx().cols
    }

    /// Returns the number of size per polynomial.
    fn size(&self) -> usize {
        self.znx().size
    }

    fn data(&self) -> &[u8] {
        &self.znx().data
    }

    fn ptr(&self) -> *mut u8 {
        self.znx().ptr
    }

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

impl ZnxBase {
    pub fn from_bytes(n: usize, rows: usize, cols: usize, size: usize, mut bytes: Vec<u8>) -> Self {
        let mut res: Self = Self::from_bytes_borrow(n, rows, cols, size, &mut bytes);
        res.data = bytes;
        res
    }

    pub fn from_bytes_borrow(n: usize, rows: usize, cols: usize, size: usize, bytes: &mut [u8]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert_eq!(n & (n - 1), 0, "n must be a power of two");
            assert!(n > 0, "n must be greater than 0");
            assert!(rows > 0, "rows must be greater than 0");
            assert!(cols > 0, "cols must be greater than 0");
            assert!(size > 0, "size must be greater than 0");
        }
        Self {
            n: n,
            rows: rows,
            cols: cols,
            size: size,
            data: Vec::new(),
            ptr: bytes.as_mut_ptr(),
        }
    }
}

pub trait ZnxAlloc<B: Backend>
where
    Self: Sized + ZnxInfos,
{
    type Scalar;
    fn new(module: &Module<B>, rows: usize, cols: usize, size: usize) -> Self {
        let bytes: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of(module, rows, cols, size));
        Self::from_bytes(module, rows, cols, size, bytes)
    }

    fn from_bytes(module: &Module<B>, rows: usize, cols: usize, size: usize, mut bytes: Vec<u8>) -> Self {
        let mut res: Self = Self::from_bytes_borrow(module, rows, cols, size, &mut bytes);
        res.znx_mut().data = bytes;
        res
    }

    fn from_bytes_borrow(module: &Module<B>, rows: usize, cols: usize, size: usize, bytes: &mut [u8]) -> Self;

    fn bytes_of(module: &Module<B>, rows: usize, cols: usize, size: usize) -> usize;
}

pub trait ZnxLayout: ZnxInfos {
    type Scalar;

    /// Returns true if the receiver is only borrowing the data.
    fn borrowing(&self) -> bool {
        self.znx().data.len() == 0
    }

    /// Returns a non-mutable pointer to the underlying coefficients array.
    fn as_ptr(&self) -> *const Self::Scalar {
        self.znx().ptr as *const Self::Scalar
    }

    /// Returns a mutable pointer to the underlying coefficients array.
    fn as_mut_ptr(&mut self) -> *mut Self::Scalar {
        self.znx_mut().ptr as *mut Self::Scalar
    }

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
            assert!(j < self.size());
        }
        let offset = self.n() * (j * self.cols() + i);
        unsafe { self.as_ptr().add(offset) }
    }

    /// Returns a mutable pointer starting at the (i, j)-th small polynomial.
    fn at_mut_ptr(&mut self, i: usize, j: usize) -> *mut Self::Scalar {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.cols());
            assert!(j < self.size());
        }
        let offset = self.n() * (j * self.cols() + i);
        unsafe { self.as_mut_ptr().add(offset) }
    }

    /// Returns non-mutable reference to the (i, j)-th small polynomial.
    fn at(&self, i: usize, j: usize) -> &[Self::Scalar] {
        unsafe { std::slice::from_raw_parts(self.at_ptr(i, j), self.n()) }
    }

    /// Returns mutable reference to the (i, j)-th small polynomial.
    fn at_mut(&mut self, i: usize, j: usize) -> &mut [Self::Scalar] {
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

use std::convert::TryFrom;
use std::num::TryFromIntError;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Shl, Shr, Sub};
pub trait IntegerType:
    Copy
    + std::fmt::Debug
    + Default
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Shr<Output = Self>
    + Shl<Output = Self>
    + AddAssign
    + TryFrom<usize, Error = TryFromIntError>
{
    const BITS: u32;
}

impl IntegerType for i64 {
    const BITS: u32 = 64;
}

impl IntegerType for i128 {
    const BITS: u32 = 128;
}

pub trait ZnxBasics: ZnxLayout
where
    Self: Sized,
    Self::Scalar: IntegerType,
{
    fn zero(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.as_mut_ptr(), 0, self.n() * size_of::<Self::Scalar>());
        }
    }

    fn zero_at(&mut self, i: usize, j: usize) {
        unsafe {
            std::ptr::write_bytes(
                self.at_mut_ptr(i, j),
                0,
                self.n() * size_of::<Self::Scalar>(),
            );
        }
    }

    fn rsh(&mut self, log_base2k: usize, k: usize, carry: &mut [u8]) {
        rsh(log_base2k, self, k, carry)
    }
}

pub fn rsh<V: ZnxBasics>(log_base2k: usize, a: &mut V, k: usize, tmp_bytes: &mut [u8])
where
    V::Scalar: IntegerType,
{
    let n: usize = a.n();
    let size: usize = a.size();
    let cols: usize = a.cols();

    #[cfg(debug_assertions)]
    {
        assert!(
            tmp_bytes.len() >= rsh_tmp_bytes::<V::Scalar>(n, cols),
            "invalid carry: carry.len()/size_ofSelf::Scalar={} < rsh_tmp_bytes({}, {})",
            tmp_bytes.len() / size_of::<V::Scalar>(),
            n,
            size,
        );
        assert_alignement(tmp_bytes.as_ptr());
    }

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
        let carry: &mut [V::Scalar] = cast_mut(tmp_bytes);

        unsafe {
            std::ptr::write_bytes(carry.as_mut_ptr(), 0, n * size_of::<V::Scalar>());
        }

        let log_base2k_t: V::Scalar = V::Scalar::try_from(log_base2k).unwrap();
        let shift: V::Scalar = V::Scalar::try_from(V::Scalar::BITS as usize - k_rem).unwrap();
        let k_rem_t: V::Scalar = V::Scalar::try_from(k_rem).unwrap();

        (steps..size).for_each(|i| {
            izip!(carry.iter_mut(), a.at_limb_mut(i).iter_mut()).for_each(|(ci, xi)| {
                *xi += *ci << log_base2k_t;
                *ci = get_base_k_carry(*xi, shift);
                *xi = (*xi - *ci) >> k_rem_t;
            });
        })
    }
}

#[inline(always)]
fn get_base_k_carry<T: IntegerType>(x: T, shift: T) -> T {
    (x << shift) >> shift
}

pub fn rsh_tmp_bytes<T: IntegerType>(n: usize, cols: usize) -> usize {
    n * cols * std::mem::size_of::<T>()
}

pub fn switch_degree<T: ZnxLayout + ZnxBasics>(b: &mut T, col_b: usize, a: &T, col_a: usize)
where
    <T as ZnxLayout>::Scalar: IntegerType,
{
    let (n_in, n_out) = (a.n(), b.n());
    let (gap_in, gap_out): (usize, usize);

    if n_in > n_out {
        (gap_in, gap_out) = (n_in / n_out, 1)
    } else {
        (gap_in, gap_out) = (1, n_out / n_in);
        b.zero();
    }

    let size: usize = min(a.size(), b.size());

    (0..size).for_each(|i| {
        izip!(
            a.at(col_a, i).iter().step_by(gap_in),
            b.at_mut(col_b, i).iter_mut().step_by(gap_out)
        )
        .for_each(|(x_in, x_out)| *x_out = *x_in);
    });
}
