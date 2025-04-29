use crate::{Backend, Module, assert_alignement, cast_mut};
use itertools::izip;
use std::cmp::{max, min};

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

    /// Returns the slice size, which is the offset between
    /// two size of the same column.
    fn sl(&self) -> usize {
        self.n() * self.cols()
    }
}

pub trait ZnxBase<B: Backend> {
    type Scalar;
    fn new(module: &Module<B>, cols: usize, size: usize) -> Self;
    fn from_bytes(module: &Module<B>, cols: usize, size: usize, bytes: &mut [u8]) -> Self;
    fn from_bytes_borrow(module: &Module<B>, cols: usize, size: usize, bytes: &mut [u8]) -> Self;
    fn bytes_of(module: &Module<B>, cols: usize, size: usize) -> usize;
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

pub fn switch_degree<T: ZnxLayout + ZnxBasics>(b: &mut T, a: &T)
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
            a.at_limb(i).iter().step_by(gap_in),
            b.at_limb_mut(i).iter_mut().step_by(gap_out)
        )
        .for_each(|(x_in, x_out)| *x_out = *x_in);
    });
}

pub fn znx_post_process_ternary_op<T: ZnxLayout + ZnxBasics, const NEGATE: bool>(c: &mut T, a: &T, b: &T)
where
    <T as ZnxLayout>::Scalar: IntegerType,
{
    #[cfg(debug_assertions)]
    {
        assert_ne!(a.as_ptr(), b.as_ptr());
        assert_ne!(b.as_ptr(), c.as_ptr());
        assert_ne!(a.as_ptr(), c.as_ptr());
    }

    let a_cols: usize = a.cols();
    let b_cols: usize = b.cols();
    let c_cols: usize = c.cols();

    let min_ab_cols: usize = min(a_cols, b_cols);
    let max_ab_cols: usize = max(a_cols, b_cols);

    // Copies shared shared cols between (c, max(a, b))
    if a_cols != b_cols {
        let mut x: &T = a;
        if a_cols < b_cols {
            x = b;
        }

        let min_size = min(c.size(), x.size());
        (min_ab_cols..min(max_ab_cols, c_cols)).for_each(|i| {
            (0..min_size).for_each(|j| {
                c.at_poly_mut(i, j).copy_from_slice(x.at_poly(i, j));
                if NEGATE {
                    c.at_poly_mut(i, j).iter_mut().for_each(|x| *x = -*x);
                }
            });
            (min_size..c.size()).for_each(|j| {
                c.zero_at(i, j);
            });
        });
    }

    // Zeroes the cols of c > max(a, b).
    if c_cols > max_ab_cols {
        (max_ab_cols..c_cols).for_each(|i| {
            (0..c.size()).for_each(|j| {
                c.zero_at(i, j);
            })
        });
    }
}

#[inline(always)]
pub fn apply_binary_op<B: Backend, T: ZnxBasics + ZnxLayout, const NEGATE: bool>(
    module: &Module<B>,
    c: &mut T,
    a: &T,
    b: &T,
    op: impl Fn(&mut [T::Scalar], &[T::Scalar], &[T::Scalar]),
) where
    <T as ZnxLayout>::Scalar: IntegerType,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), module.n());
        assert_eq!(b.n(), module.n());
        assert_eq!(c.n(), module.n());
        assert_ne!(a.as_ptr(), b.as_ptr());
    }
    let a_cols: usize = a.cols();
    let b_cols: usize = b.cols();
    let c_cols: usize = c.cols();
    let min_ab_cols: usize = min(a_cols, b_cols);
    let min_cols: usize = min(c_cols, min_ab_cols);
    // Applies over shared cols between (a, b, c)
    (0..min_cols).for_each(|i| op(c.at_poly_mut(i, 0), a.at_poly(i, 0), b.at_poly(i, 0)));
    // Copies/Negates/Zeroes the remaining cols if op is not inplace.
    if c.as_ptr() != a.as_ptr() && c.as_ptr() != b.as_ptr() {
        znx_post_process_ternary_op::<T, NEGATE>(c, a, b);
    }
}

#[inline(always)]
pub fn apply_unary_op<B: Backend, T: ZnxBasics + ZnxLayout>(
    module: &Module<B>,
    b: &mut T,
    a: &T,
    op: impl Fn(&mut [T::Scalar], &[T::Scalar]),
) where
    <T as ZnxLayout>::Scalar: IntegerType,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), module.n());
        assert_eq!(b.n(), module.n());
    }
    let a_cols: usize = a.cols();
    let b_cols: usize = b.cols();
    let min_cols: usize = min(a_cols, b_cols);
    // Applies over the shared cols between (a, b)
    (0..min_cols).for_each(|i| op(b.at_poly_mut(i, 0), a.at_poly(i, 0)));
    // Zeroes the remaining cols of b.
    (min_cols..b_cols).for_each(|i| (0..b.size()).for_each(|j| b.zero_at(i, j)));
}
