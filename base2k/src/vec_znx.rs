use crate::cast_mut;
use crate::ffi::vec_znx;
use crate::ffi::znx;
use crate::ffi::znx::znx_zero_i64_ref;
use crate::{alias_mut_slice_to_vec, alloc_aligned};
use crate::{Infos, Module};
use itertools::izip;
use std::cmp::min;

pub trait VecZnxVec {
    fn dblptr(&self) -> Vec<&[i64]>;
    fn dblptr_mut(&mut self) -> Vec<&mut [i64]>;
}

impl<T: VecZnxCommon> VecZnxVec for Vec<T> {
    fn dblptr(&self) -> Vec<&[i64]> {
        self.iter().map(|v| v.raw()).collect()
    }

    fn dblptr_mut(&mut self) -> Vec<&mut [i64]> {
        self.iter_mut().map(|v| v.raw_mut()).collect()
    }
}

pub trait VecZnxApi: AsRef<Self> + AsMut<Self> {
    type Owned: VecZnxCommon;

    fn from_bytes(n: usize, cols: usize, bytes: &mut [u8]) -> Self::Owned;

    /// Returns the minimum size of the [u8] array required to assign a
    /// new backend array.
    fn bytes_of(n: usize, cols: usize) -> usize;

    /// Copy the data of a onto self.
    fn copy_from<A: VecZnxCommon, B: VecZnxCommon>(&mut self, a: &A)
    where
        Self: AsMut<B>;

    /// Returns the backing array.
    fn raw(&self) -> &[i64];

    /// Returns the mutable backing array.
    fn raw_mut(&mut self) -> &mut [i64];

    /// Returns a non-mutable pointer to the backing array.
    fn as_ptr(&self) -> *const i64;

    /// Returns a mutable pointer to the backing array.
    fn as_mut_ptr(&mut self) -> *mut i64;

    /// Returns a non-mutable reference to the i-th cols.
    fn at(&self, i: usize) -> &[i64];

    /// Returns a mutable reference to the i-th cols .
    fn at_mut(&mut self, i: usize) -> &mut [i64];

    /// Returns a non-mutable pointer to the i-th cols.
    fn at_ptr(&self, i: usize) -> *const i64;

    /// Returns a mutable pointer to the i-th cols.
    fn at_mut_ptr(&mut self, i: usize) -> *mut i64;

    /// Zeroes the backing array.
    fn zero(&mut self);

    /// Normalization: propagates carry and ensures each coefficients
    /// falls into the range [-2^{K-1}, 2^{K-1}].
    fn normalize(&mut self, log_base2k: usize, carry: &mut [u8]);

    /// Right shifts the coefficients by k bits.
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: the base two logarithm of the coefficients decomposition.
    /// * `k`: the shift amount.
    /// * `carry`: scratch space of size at least equal to self.n() * self.cols() << 3.
    ///
    /// # Panics
    ///
    /// The method will panic if carry.len() < self.n() * self.cols() << 3.
    fn rsh(&mut self, log_base2k: usize, k: usize, carry: &mut [u8]);

    /// If self.n() > a.n(): Extracts X^{i*self.n()/a.n()} -> X^{i}.
    /// If self.n() < a.n(): Extracts X^{i} -> X^{i*a.n()/self.n()}.
    ///
    /// # Arguments
    ///
    /// * `a`: the receiver polynomial in which the extracted coefficients are stored.
    fn switch_degree<A: VecZnxCommon, B: VecZnxCommon>(&self, a: &mut A)
    where
        Self: AsRef<B>;

    fn print(&self, cols: usize, n: usize);
}

pub fn bytes_of_vec_znx(n: usize, cols: usize) -> usize {
    n * cols * 8
}

pub struct VecZnxBorrow {
    pub n: usize,
    pub cols: usize,
    pub data: *mut i64,
}

impl AsMut<VecZnxBorrow> for VecZnxBorrow {
    fn as_mut(&mut self) -> &mut VecZnxBorrow {
        self
    }
}

impl AsRef<VecZnxBorrow> for VecZnxBorrow {
    fn as_ref(&self) -> &VecZnxBorrow {
        self
    }
}

impl VecZnxCommon for VecZnxBorrow {}

impl VecZnxApi for VecZnxBorrow {
    type Owned = VecZnxBorrow;

    /// Returns a new struct implementing [VecZnxBorrow] with the provided data as backing array.
    ///
    /// The struct will *NOT* take ownership of buf[..[VecZnx::bytes_of]]
    ///
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [VecZnx::bytes_of].
    fn from_bytes(n: usize, cols: usize, bytes: &mut [u8]) -> Self::Owned {
        let size = Self::bytes_of(n, cols);
        assert!(
            bytes.len() >= size,
            "invalid buffer: buf.len()={} < self.buffer_size(n={}, cols={})={}",
            bytes.len(),
            n,
            cols,
            size
        );
        VecZnxBorrow {
            n: n,
            cols: cols,
            data: cast_mut(&mut bytes[..size]).as_mut_ptr(),
        }
    }

    fn bytes_of(n: usize, cols: usize) -> usize {
        bytes_of_vec_znx(n, cols)
    }

    fn copy_from<A: VecZnxCommon, B: VecZnxCommon>(&mut self, a: &A)
    where
        Self: AsMut<B>,
    {
        copy_vec_znx_from::<A, B>(self.as_mut(), a);
    }

    fn as_ptr(&self) -> *const i64 {
        self.data
    }

    fn as_mut_ptr(&mut self) -> *mut i64 {
        self.data
    }

    fn raw(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.data, self.n * self.cols) }
    }

    fn raw_mut(&mut self) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.n * self.cols) }
    }

    fn at(&self, i: usize) -> &[i64] {
        let n: usize = self.n();
        &self.raw()[n * i..n * (i + 1)]
    }

    fn at_mut(&mut self, i: usize) -> &mut [i64] {
        let n: usize = self.n();
        &mut self.raw_mut()[n * i..n * (i + 1)]
    }

    fn at_ptr(&self, i: usize) -> *const i64 {
        self.data.wrapping_add(self.n * i)
    }

    fn at_mut_ptr(&mut self, i: usize) -> *mut i64 {
        self.data.wrapping_add(self.n * i)
    }

    fn zero(&mut self) {
        unsafe {
            znx_zero_i64_ref((self.n * self.cols) as u64, self.data);
        }
    }

    fn normalize(&mut self, log_base2k: usize, carry: &mut [u8]) {
        normalize(log_base2k, self, carry)
    }

    fn rsh(&mut self, log_base2k: usize, k: usize, carry: &mut [u8]) {
        rsh(log_base2k, self, k, carry)
    }

    fn switch_degree<A: VecZnxCommon, B: VecZnxCommon>(&self, a: &mut A)
    where
        Self: AsRef<B>,
    {
        switch_degree(a, self.as_ref());
    }

    fn print(&self, cols: usize, n: usize) {
        (0..cols).for_each(|i| println!("{}: {:?}", i, &self.at(i)[..n]))
    }
}

impl VecZnxCommon for VecZnx {}

impl VecZnxApi for VecZnx {
    type Owned = VecZnx;

    /// Returns a new struct implementing [VecZnx] with the provided data as backing array.
    ///
    /// The struct will take ownership of buf[..[VecZnx::bytes_of]]
    ///
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [VecZnx::bytes_of].
    fn from_bytes(n: usize, cols: usize, buf: &mut [u8]) -> Self::Owned {
        let size = Self::bytes_of(n, cols);
        assert!(
            buf.len() >= size,
            "invalid buffer: buf.len()={} < self.buffer_size(n={}, cols={})={}",
            buf.len(),
            n,
            cols,
            size
        );

        VecZnx {
            n: n,
            data: alias_mut_slice_to_vec(cast_mut(&mut buf[..size])),
        }
    }

    fn bytes_of(n: usize, cols: usize) -> usize {
        bytes_of_vec_znx(n, cols)
    }

    fn copy_from<A: VecZnxCommon, B: VecZnxCommon>(&mut self, a: &A)
    where
        Self: AsMut<B>,
    {
        copy_vec_znx_from(self.as_mut(), a);
    }

    fn raw(&self) -> &[i64] {
        &self.data
    }

    fn raw_mut(&mut self) -> &mut [i64] {
        &mut self.data
    }

    fn as_ptr(&self) -> *const i64 {
        self.data.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut i64 {
        self.data.as_mut_ptr()
    }

    fn at(&self, i: usize) -> &[i64] {
        let n: usize = self.n();
        &self.raw()[n * i..n * (i + 1)]
    }

    fn at_mut(&mut self, i: usize) -> &mut [i64] {
        let n: usize = self.n();
        &mut self.raw_mut()[n * i..n * (i + 1)]
    }

    fn at_ptr(&self, i: usize) -> *const i64 {
        &self.data[i * self.n] as *const i64
    }

    fn at_mut_ptr(&mut self, i: usize) -> *mut i64 {
        &mut self.data[i * self.n] as *mut i64
    }

    fn zero(&mut self) {
        unsafe { znx::znx_zero_i64_ref(self.data.len() as u64, self.data.as_mut_ptr()) }
    }

    fn normalize(&mut self, log_base2k: usize, carry: &mut [u8]) {
        normalize(log_base2k, self, carry)
    }

    fn rsh(&mut self, log_base2k: usize, k: usize, carry: &mut [u8]) {
        rsh(log_base2k, self, k, carry)
    }

    fn switch_degree<A: VecZnxCommon, B: VecZnxCommon>(&self, a: &mut A)
    where
        Self: AsRef<B>,
    {
        switch_degree(a, self.as_ref())
    }

    fn print(&self, cols: usize, n: usize) {
        (0..cols).for_each(|i| println!("{}: {:?}", i, &self.at(i)[..n]))
    }
}

/// [VecZnx] represents a vector of small norm polynomials of Zn\[X\] with [i64] coefficients.
/// A [VecZnx] is composed of multiple Zn\[X\] polynomials stored in a single contiguous array
/// in the memory.
#[derive(Clone)]
pub struct VecZnx {
    /// Polynomial degree.
    pub n: usize,
    /// Polynomial coefficients, as a contiguous array. Each col is equally spaced by n.
    pub data: Vec<i64>,
}

impl AsMut<VecZnx> for VecZnx {
    fn as_mut(&mut self) -> &mut VecZnx {
        self
    }
}

impl AsRef<VecZnx> for VecZnx {
    fn as_ref(&self) -> &VecZnx {
        self
    }
}

/// Copies the coefficients of `a` on the receiver.
/// Copy is done with the minimum size matching both backing arrays.
pub fn copy_vec_znx_from<A: VecZnxCommon, B: VecZnxCommon>(b: &mut B, a: &A) {
    let data_a: &[i64] = a.raw();
    let data_b: &mut [i64] = b.raw_mut();
    let size = min(data_b.len(), data_a.len());
    data_b[..size].copy_from_slice(&data_a[..size])
}

impl VecZnx {
    /// Allocates a new [VecZnx] composed of #cols polynomials of Z\[X\].
    pub fn new(n: usize, cols: usize) -> Self {
        Self {
            n: n,
            data: alloc_aligned::<i64>(n * cols, 64),
        }
    }

    /// Truncates the precision of the [VecZnx] by k bits.
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: the base two logarithm of the coefficients decomposition.
    /// * `k`: the number of bits of precision to drop.
    pub fn trunc_pow2(&mut self, log_base2k: usize, k: usize) {
        if k == 0 {
            return;
        }

        self.data
            .truncate((self.cols() - k / log_base2k) * self.n());

        let k_rem: usize = k % log_base2k;

        if k_rem != 0 {
            let mask: i64 = ((1 << (log_base2k - k_rem - 1)) - 1) << k_rem;
            self.at_mut(self.cols() - 1)
                .iter_mut()
                .for_each(|x: &mut i64| *x &= mask)
        }
    }
}

pub fn switch_degree<A: VecZnxCommon, B: VecZnxCommon>(b: &mut B, a: &A) {
    let (n_in, n_out) = (a.n(), b.n());
    let (gap_in, gap_out): (usize, usize);

    if n_in > n_out {
        (gap_in, gap_out) = (n_in / n_out, 1)
    } else {
        (gap_in, gap_out) = (1, n_out / n_in);
        b.zero();
    }

    let cols = min(a.cols(), b.cols());

    (0..cols).for_each(|i| {
        izip!(
            a.at(i).iter().step_by(gap_in),
            b.at_mut(i).iter_mut().step_by(gap_out)
        )
        .for_each(|(x_in, x_out)| *x_out = *x_in);
    });
}

fn normalize<T: VecZnxCommon>(log_base2k: usize, a: &mut T, carry: &mut [u8]) {
    let n: usize = a.n();

    assert!(
        carry.len() >= n * 8,
        "invalid carry: carry.len()={} < self.n()={}",
        carry.len(),
        n
    );

    let carry_i64: &mut [i64] = cast_mut(carry);

    unsafe {
        znx::znx_zero_i64_ref(n as u64, carry_i64.as_mut_ptr());
        (0..a.cols()).rev().for_each(|i| {
            znx::znx_normalize(
                n as u64,
                log_base2k as u64,
                a.at_mut_ptr(i),
                carry_i64.as_mut_ptr(),
                a.at_mut_ptr(i),
                carry_i64.as_mut_ptr(),
            )
        });
    }
}

pub fn rsh<T: VecZnxCommon>(log_base2k: usize, a: &mut T, k: usize, carry: &mut [u8]) {
    let n: usize = a.n();

    assert!(
        carry.len() >> 3 >= n,
        "invalid carry: carry.len()/8={} < self.n()={}",
        carry.len() >> 3,
        n
    );

    let cols: usize = a.cols();
    let cols_steps: usize = k / log_base2k;

    a.raw_mut().rotate_right(n * cols_steps);
    unsafe {
        znx::znx_zero_i64_ref((n * cols_steps) as u64, a.as_mut_ptr());
    }

    let k_rem = k % log_base2k;

    if k_rem != 0 {
        let carry_i64: &mut [i64] = cast_mut(carry);

        unsafe {
            znx::znx_zero_i64_ref(n as u64, carry_i64.as_mut_ptr());
        }

        let mask: i64 = (1 << k_rem) - 1;
        let log_base2k: usize = log_base2k;

        (cols_steps..cols).for_each(|i| {
            izip!(carry_i64.iter_mut(), a.at_mut(i).iter_mut()).for_each(|(ci, xi)| {
                *xi += *ci << log_base2k;
                *ci = *xi & mask;
                *xi /= 1 << k_rem;
            });
        })
    }
}

pub trait VecZnxCommon: VecZnxApi + Infos {}

pub trait VecZnxOps {
    /// Allocates a new [VecZnx].
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of cols.
    fn new_vec_znx(&self, cols: usize) -> VecZnx;

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnx] through [VecZnx::from_bytes].
    fn bytes_of_vec_znx(&self, cols: usize) -> usize;

    /// c <- a + b.
    fn vec_znx_add<A: VecZnxCommon, B: VecZnxCommon, C: VecZnxCommon>(
        &self,
        c: &mut C,
        a: &A,
        b: &B,
    );

    /// b <- b + a.
    fn vec_znx_add_inplace<A: VecZnxCommon, B: VecZnxCommon>(&self, b: &mut B, a: &A);

    /// c <- a - b.
    fn vec_znx_sub<A: VecZnxCommon, B: VecZnxCommon, C: VecZnxCommon>(
        &self,
        c: &mut C,
        a: &A,
        b: &B,
    );

    /// b <- b - a.
    fn vec_znx_sub_inplace<A: VecZnxCommon, B: VecZnxCommon>(&self, b: &mut B, a: &A);

    /// b <- -a.
    fn vec_znx_negate<A: VecZnxCommon, B: VecZnxCommon>(&self, b: &mut B, a: &A);

    /// b <- -b.
    fn vec_znx_negate_inplace<A: VecZnxCommon>(&self, a: &mut A);

    /// b <- a * X^k (mod X^{n} + 1)
    fn vec_znx_rotate<A: VecZnxCommon, B: VecZnxCommon>(&self, k: i64, b: &mut B, a: &A);

    /// a <- a * X^k (mod X^{n} + 1)
    fn vec_znx_rotate_inplace<A: VecZnxCommon>(&self, k: i64, a: &mut A);

    /// b <- phi_k(a) where phi_k: X^i -> X^{i*k} (mod (X^{n} + 1))
    fn vec_znx_automorphism<A: VecZnxCommon, B: VecZnxCommon>(
        &self,
        k: i64,
        b: &mut B,
        a: &A,
        a_cols: usize,
    );

    /// a <- phi_k(a) where phi_k: X^i -> X^{i*k} (mod (X^{n} + 1))
    fn vec_znx_automorphism_inplace<A: VecZnxCommon>(&self, k: i64, a: &mut A, a_cols: usize);

    /// Splits b into subrings and copies them them into a.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of b have the same ring degree
    /// and that b.n() * b.len() <= a.n()
    fn vec_znx_split<A: VecZnxCommon, B: VecZnxCommon, C: VecZnxCommon>(
        &self,
        b: &mut Vec<B>,
        a: &A,
        buf: &mut C,
    );

    /// Merges the subrings a into b.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of a have the same ring degree
    /// and that a.n() * a.len() <= b.n()
    fn vec_znx_merge<A: VecZnxCommon, B: VecZnxCommon>(&self, b: &mut B, a: &Vec<A>);
}

impl VecZnxOps for Module {
    fn new_vec_znx(&self, cols: usize) -> VecZnx {
        VecZnx::new(self.n(), cols)
    }

    fn bytes_of_vec_znx(&self, cols: usize) -> usize {
        self.n() * cols * 8
    }

    // c <- a + b
    fn vec_znx_add<A: VecZnxCommon, B: VecZnxCommon, C: VecZnxCommon>(
        &self,
        c: &mut C,
        a: &A,
        b: &B,
    ) {
        unsafe {
            vec_znx::vec_znx_add(
                self.0,
                c.as_mut_ptr(),
                c.cols() as u64,
                c.n() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
                b.as_ptr(),
                b.cols() as u64,
                b.n() as u64,
            )
        }
    }

    // b <- a + b
    fn vec_znx_add_inplace<A: VecZnxCommon, B: VecZnxCommon>(&self, b: &mut B, a: &A) {
        unsafe {
            vec_znx::vec_znx_add(
                self.0,
                b.as_mut_ptr(),
                b.cols() as u64,
                b.n() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
                b.as_ptr(),
                b.cols() as u64,
                b.n() as u64,
            )
        }
    }

    // c <- a + b
    fn vec_znx_sub<A: VecZnxCommon, B: VecZnxCommon, C: VecZnxCommon>(
        &self,
        c: &mut C,
        a: &A,
        b: &B,
    ) {
        unsafe {
            vec_znx::vec_znx_sub(
                self.0,
                c.as_mut_ptr(),
                c.cols() as u64,
                c.n() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
                b.as_ptr(),
                b.cols() as u64,
                b.n() as u64,
            )
        }
    }

    // b <- a + b
    fn vec_znx_sub_inplace<A: VecZnxCommon, B: VecZnxCommon>(&self, b: &mut B, a: &A) {
        unsafe {
            vec_znx::vec_znx_sub(
                self.0,
                b.as_mut_ptr(),
                b.cols() as u64,
                b.n() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
                b.as_ptr(),
                b.cols() as u64,
                b.n() as u64,
            )
        }
    }

    fn vec_znx_negate<A: VecZnxCommon, B: VecZnxCommon>(&self, b: &mut B, a: &A) {
        unsafe {
            vec_znx::vec_znx_negate(
                self.0,
                b.as_mut_ptr(),
                b.cols() as u64,
                b.n() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
            )
        }
    }

    fn vec_znx_negate_inplace<A: VecZnxCommon>(&self, a: &mut A) {
        unsafe {
            vec_znx::vec_znx_negate(
                self.0,
                a.as_mut_ptr(),
                a.cols() as u64,
                a.n() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
            )
        }
    }

    fn vec_znx_rotate<A: VecZnxCommon, B: VecZnxCommon>(&self, k: i64, b: &mut B, a: &A) {
        unsafe {
            vec_znx::vec_znx_rotate(
                self.0,
                k,
                b.as_mut_ptr(),
                b.cols() as u64,
                b.n() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
            )
        }
    }

    fn vec_znx_rotate_inplace<A: VecZnxCommon>(&self, k: i64, a: &mut A) {
        unsafe {
            vec_znx::vec_znx_rotate(
                self.0,
                k,
                a.as_mut_ptr(),
                a.cols() as u64,
                a.n() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
            )
        }
    }

    /// Maps X^i to X^{ik} mod X^{n}+1. The mapping is applied independently on each cols.
    ///
    /// # Arguments
    ///
    /// * `a`: input.
    /// * `b`: output.
    /// * `k`: the power to which to map each coefficients.
    /// * `a_cols`: the number of a_cols on which to apply the mapping.
    ///
    /// # Panics
    ///
    /// The method will panic if the argument `a` is greater than `a.cols()`.
    ///
    /// # Example
    /// ```
    /// use base2k::{Module, FFT64, VecZnx, Encoding, Infos, VecZnxApi, VecZnxOps};
    /// use itertools::izip;
    ///
    /// let n: usize = 8; // polynomial degree
    /// let module = Module::new::<FFT64>(n);
    /// let mut a: VecZnx = module.new_vec_znx(2);
    /// let mut b: VecZnx = module.new_vec_znx(2);
    /// let mut c: VecZnx = module.new_vec_znx(2);
    ///
    /// (0..a.cols()).for_each(|i|{
    ///     a.at_mut(i).iter_mut().enumerate().for_each(|(i, x)|{
    ///         *x = i as i64
    ///     })
    /// });
    ///
    /// module.vec_znx_automorphism(-1, &mut b, &a, 1); // X^i -> X^(-i)
    /// let col = c.at_mut(0);
    /// (1..col.len()).for_each(|i|{
    ///     col[n-i] = -(i as i64)
    /// });
    /// izip!(b.data.iter(), c.data.iter()).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
    /// ```
    fn vec_znx_automorphism<A: VecZnxCommon, B: VecZnxCommon>(
        &self,
        k: i64,
        b: &mut B,
        a: &A,
        a_cols: usize,
    ) {
        assert_eq!(a.n(), self.n());
        assert_eq!(b.n(), self.n());
        assert!(a.cols() >= a_cols);
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.0,
                k,
                b.as_mut_ptr(),
                b.cols() as u64,
                b.n() as u64,
                a.as_ptr(),
                a_cols as u64,
                a.n() as u64,
            );
        }
    }

    /// Maps X^i to X^{ik} mod X^{n}+1. The mapping is applied independently on each cols.
    ///
    /// # Arguments
    ///
    /// * `a`: input and output.
    /// * `k`: the power to which to map each coefficients.
    /// * `a_cols`: the number of cols on which to apply the mapping.
    ///
    /// # Panics
    ///
    /// The method will panic if the argument `cols` is greater than `self.cols()`.
    ///
    /// # Example
    /// ```
    /// use base2k::{Module, FFT64, VecZnx, Encoding, Infos, VecZnxApi, VecZnxOps};
    /// use itertools::izip;
    ///
    /// let n: usize = 8; // polynomial degree
    /// let module = Module::new::<FFT64>(n);
    /// let mut a: VecZnx = VecZnx::new(n, 2);
    /// let mut b: VecZnx = VecZnx::new(n, 2);
    ///
    /// (0..a.cols()).for_each(|i|{
    ///     a.at_mut(i).iter_mut().enumerate().for_each(|(i, x)|{
    ///         *x = i as i64
    ///     })
    /// });
    ///
    /// module.vec_znx_automorphism_inplace(-1, &mut a, 1); // X^i -> X^(-i)
    /// let col = b.at_mut(0);
    /// (1..col.len()).for_each(|i|{
    ///     col[n-i] = -(i as i64)
    /// });
    /// izip!(a.data.iter(), b.data.iter()).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
    /// ```
    fn vec_znx_automorphism_inplace<A: VecZnxCommon>(&self, k: i64, a: &mut A, a_cols: usize) {
        assert_eq!(a.n(), self.n());
        assert!(a.cols() >= a_cols);
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.0,
                k,
                a.as_mut_ptr(),
                a.cols() as u64,
                a.n() as u64,
                a.as_ptr(),
                a_cols as u64,
                a.n() as u64,
            );
        }
    }

    fn vec_znx_split<A: VecZnxCommon, B: VecZnxCommon, C: VecZnxCommon>(
        &self,
        b: &mut Vec<B>,
        a: &A,
        buf: &mut C,
    ) {
        let (n_in, n_out) = (a.n(), b[0].n());

        assert!(
            n_out < n_in,
            "invalid a: output ring degree should be smaller"
        );
        b[1..].iter().for_each(|bi| {
            assert_eq!(
                bi.n(),
                n_out,
                "invalid input a: all VecZnx must have the same degree"
            )
        });

        b.iter_mut().enumerate().for_each(|(i, bi)| {
            if i == 0 {
                switch_degree(bi, a);
                self.vec_znx_rotate(-1, buf, a);
            } else {
                switch_degree(bi, buf);
                self.vec_znx_rotate_inplace(-1, buf);
            }
        })
    }

    fn vec_znx_merge<A: VecZnxCommon, B: VecZnxCommon>(&self, b: &mut B, a: &Vec<A>) {
        let (n_in, n_out) = (b.n(), a[0].n());

        assert!(
            n_out < n_in,
            "invalid a: output ring degree should be smaller"
        );
        a[1..].iter().for_each(|ai| {
            assert_eq!(
                ai.n(),
                n_out,
                "invalid input a: all VecZnx must have the same degree"
            )
        });

        a.iter().enumerate().for_each(|(_, ai)| {
            switch_degree(b, ai);
            self.vec_znx_rotate_inplace(-1, b);
        });

        self.vec_znx_rotate_inplace(a.len() as i64, b);
    }
}
