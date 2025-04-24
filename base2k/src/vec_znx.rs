use crate::cast_mut;
use crate::ffi::vec_znx;
use crate::ffi::znx;
use crate::{Infos, Module};
use crate::{alloc_aligned, assert_alignement};
use itertools::izip;
use std::cmp::min;

/// [VecZnx] represents collection of contiguously stacked vector of small norm polynomials of
/// Zn\[X\] with [i64] coefficients.
/// A [VecZnx] is composed of multiple Zn\[X\] polynomials stored in a single contiguous array
/// in the memory.
///
/// # Example
///
/// Given 3 polynomials (a, b, c) of Zn\[X\], each with 4 columns, then the memory
/// layout is: `[a0, b0, c0, a1, b1, c1, a2, b2, c2, a3, b3, c3]`, where ai, bi, ci
/// are small polynomials of Zn\[X\].
#[derive(Clone)]
pub struct VecZnx {
    /// Polynomial degree.
    pub n: usize,

    /// Stack size
    pub size: usize,

    /// Number of columns.
    pub cols: usize,

    /// Polynomial coefficients, as a contiguous array. Each col is equally spaced by n.
    pub data: Vec<i64>,

    /// Pointer to data (data can be enpty if [VecZnx] borrows space instead of owning it).
    pub ptr: *mut i64,
}

pub fn bytes_of_vec_znx(n: usize, size: usize, cols: usize) -> usize {
    n * size * cols * 8
}

impl VecZnx {
    /// Returns a new struct implementing [VecZnx] with the provided data as backing array.
    ///
    /// The struct will take ownership of buf[..[VecZnx::bytes_of]]
    ///
    /// User must ensure that data is properly alligned and that
    /// the size of data is equal to [VecZnx::bytes_of].
    pub fn from_bytes(n: usize, size: usize, cols: usize, bytes: &mut [u8]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(size > 0);
            assert_eq!(bytes.len(), Self::bytes_of(n, size, cols));
            assert_alignement(bytes.as_ptr());
        }
        unsafe {
            let bytes_i64: &mut [i64] = cast_mut::<u8, i64>(bytes);
            let ptr: *mut i64 = bytes_i64.as_mut_ptr();
            VecZnx {
                n: n,
                size: size,
                cols: cols,
                data: Vec::from_raw_parts(ptr, bytes.len(), bytes.len()),
                ptr: ptr,
            }
        }
    }

    pub fn from_bytes_borrow(n: usize, size: usize, cols: usize, bytes: &mut [u8]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(size > 0);
            assert!(bytes.len() >= Self::bytes_of(n, size, cols));
            assert_alignement(bytes.as_ptr());
        }
        VecZnx {
            n: n,
            size: size,
            cols: cols,
            data: Vec::new(),
            ptr: bytes.as_mut_ptr() as *mut i64,
        }
    }

    pub fn bytes_of(n: usize, size: usize, cols: usize) -> usize {
        bytes_of_vec_znx(n, size, cols)
    }

    pub fn copy_from(&mut self, a: &VecZnx) {
        copy_vec_znx_from(self, a);
    }

    pub fn borrowing(&self) -> bool {
        self.data.len() == 0
    }

    /// TODO: when SML refactoring is done, move this to the [Infos] trait.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Total size is [VecZnx::n()] * [VecZnx::size()] * [VecZnx::cols()].
    pub fn raw(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.n * self.size * self.cols) }
    }

    /// Returns a reference to backend slice of the receiver.
    /// Total size is [VecZnx::n()] * [VecZnx::size()] * [VecZnx::cols()].
    pub fn raw_mut(&mut self) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.n * self.size * self.cols) }
    }

    /// Returns a non-mutable pointer to the backedn slice of the receiver.
    pub fn as_ptr(&self) -> *const i64 {
        self.ptr
    }

    /// Returns a mutable pointer to the backedn slice of the receiver.
    pub fn as_mut_ptr(&mut self) -> *mut i64 {
        self.ptr
    }

    /// Returns a non-mutable pointer starting a the j-th column.
    pub fn at_ptr(&self, i: usize) -> *const i64 {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.cols);
        }
        let offset: usize = self.n * self.size * i;
        self.ptr.wrapping_add(offset)
    }

    /// Returns non-mutable reference to the ith-column.
    /// The slice contains [VecZnx::size()] small polynomials, each of [VecZnx::n()] coefficients.
    pub fn at(&self, i: usize) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.at_ptr(i), self.n * self.size) }
    }

    /// Returns a non-mutable pointer starting a the j-th column of the i-th polynomial.
    pub fn at_poly_ptr(&self, i: usize, j: usize) -> *const i64 {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.size);
            assert!(j < self.cols);
        }
        let offset: usize = self.n * (self.size * j + i);
        self.ptr.wrapping_add(offset)
    }

    /// Returns non-mutable reference to the j-th column of the i-th polynomial.
    /// The slice contains one small polynomial of [VecZnx::n()] coefficients.
    pub fn at_poly(&self, i: usize, j: usize) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.at_poly_ptr(i, j), self.n) }
    }

    /// Returns a mutable pointer starting a the j-th column.
    pub fn at_mut_ptr(&self, i: usize) -> *mut i64 {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.cols);
        }
        let offset: usize = self.n * self.size * i;
        self.ptr.wrapping_add(offset)
    }

    /// Returns mutable reference to the ith-column.
    /// The slice contains [VecZnx::size()] small polynomials, each of [VecZnx::n()] coefficients.
    pub fn at_mut(&mut self, i: usize) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.at_mut_ptr(i), self.n * self.size) }
    }

    /// Returns a mutable pointer starting a the j-th column of the i-th polynomial.
    pub fn at_poly_mut_ptr(&mut self, i: usize, j: usize) -> *mut i64 {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.size);
            assert!(j < self.cols);
        }

        let offset: usize = self.n * (self.size * j + i);
        self.ptr.wrapping_add(offset)
    }

    /// Returns mutable reference to the j-th column of the i-th polynomial.
    /// The slice contains one small polynomial of [VecZnx::n()] coefficients.
    pub fn at_poly_mut(&mut self, i: usize, j: usize) -> &mut [i64] {
        let ptr: *mut i64 = self.at_poly_mut_ptr(i, j);
        unsafe { std::slice::from_raw_parts_mut(ptr, self.n) }
    }

    pub fn zero(&mut self) {
        unsafe { znx::znx_zero_i64_ref((self.n * self.cols * self.size) as u64, self.ptr) }
    }

    pub fn normalize(&mut self, log_base2k: usize, carry: &mut [u8]) {
        normalize(log_base2k, self, carry)
    }

    pub fn rsh(&mut self, log_base2k: usize, k: usize, carry: &mut [u8]) {
        rsh(log_base2k, self, k, carry)
    }

    pub fn switch_degree(&self, a: &mut VecZnx) {
        switch_degree(a, self)
    }

    pub fn print(&self, poly: usize, cols: usize, n: usize) {
        (0..cols).for_each(|i| println!("{}: {:?}", i, &self.at_poly(poly, i)[..n]))
    }
}

impl Infos for VecZnx {
    /// Returns the base 2 logarithm of the [VecZnx] degree.
    fn log_n(&self) -> usize {
        (usize::BITS - (self.n - 1).leading_zeros()) as _
    }

    /// Returns the [VecZnx] degree.
    fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of cols of the [VecZnx].
    fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the number of rows of the [VecZnx].
    fn rows(&self) -> usize {
        1
    }
}

/// Copies the coefficients of `a` on the receiver.
/// Copy is done with the minimum size matching both backing arrays.
pub fn copy_vec_znx_from(b: &mut VecZnx, a: &VecZnx) {
    let data_a: &[i64] = a.raw();
    let data_b: &mut [i64] = b.raw_mut();
    let size = min(data_b.len(), data_a.len());
    data_b[..size].copy_from_slice(&data_a[..size])
}

impl VecZnx {
    /// Allocates a new [VecZnx] composed of #cols polynomials of Z\[X\].
    pub fn new(n: usize, size: usize, cols: usize) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(n > 0);
            assert!(n & (n - 1) == 0);
            assert!(size > 0);
            assert!(cols > 0);
        }
        let mut data: Vec<i64> = alloc_aligned::<i64>(n * size * cols);
        let ptr: *mut i64 = data.as_mut_ptr();
        Self {
            n: n,
            size: size,
            cols: cols,
            data: data,
            ptr: ptr,
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

        if !self.borrowing() {
            self.data
                .truncate((self.cols() - k / log_base2k) * self.n() * self.size());
        }

        self.cols -= k / log_base2k;

        let k_rem: usize = k % log_base2k;

        if k_rem != 0 {
            let mask: i64 = ((1 << (log_base2k - k_rem - 1)) - 1) << k_rem;
            self.at_mut(self.cols() - 1)
                .iter_mut()
                .for_each(|x: &mut i64| *x &= mask)
        }
    }
}

pub fn switch_degree(b: &mut VecZnx, a: &VecZnx) {
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

fn normalize_tmp_bytes(n: usize, size: usize) -> usize {
    n * size * std::mem::size_of::<i64>()
}

fn normalize(log_base2k: usize, a: &mut VecZnx, tmp_bytes: &mut [u8]) {
    let n: usize = a.n();
    let size: usize = a.size();

    debug_assert!(
        tmp_bytes.len() >= normalize_tmp_bytes(n, size),
        "invalid tmp_bytes: tmp_bytes.len()={} < normalize_tmp_bytes({}, {})",
        tmp_bytes.len(),
        n,
        size,
    );
    #[cfg(debug_assertions)]
    {
        assert_alignement(tmp_bytes.as_ptr())
    }

    let carry_i64: &mut [i64] = cast_mut(tmp_bytes);

    unsafe {
        znx::znx_zero_i64_ref(n as u64, carry_i64.as_mut_ptr());
        (0..a.cols()).rev().for_each(|i| {
            znx::znx_normalize(
                (n * size) as u64,
                log_base2k as u64,
                a.at_mut_ptr(i),
                carry_i64.as_mut_ptr(),
                a.at_mut_ptr(i),
                carry_i64.as_mut_ptr(),
            )
        });
    }
}

pub fn rsh_tmp_bytes(n: usize, size: usize) -> usize {
    n * size * std::mem::size_of::<i64>()
}

pub fn rsh(log_base2k: usize, a: &mut VecZnx, k: usize, tmp_bytes: &mut [u8]) {
    let n: usize = a.n();
    let size: usize = a.size();

    #[cfg(debug_assertions)]
    {
        assert!(
            tmp_bytes.len() >= rsh_tmp_bytes(n, size),
            "invalid carry: carry.len()/8={} < rsh_tmp_bytes({}, {})",
            tmp_bytes.len() >> 3,
            n,
            size,
        );
        assert_alignement(tmp_bytes.as_ptr());
    }

    let cols: usize = a.cols();
    let cols_steps: usize = k / log_base2k;

    a.raw_mut().rotate_right(n * size * cols_steps);
    unsafe {
        znx::znx_zero_i64_ref((n * size * cols_steps) as u64, a.as_mut_ptr());
    }

    let k_rem = k % log_base2k;

    if k_rem != 0 {
        let carry_i64: &mut [i64] = cast_mut(tmp_bytes);

        unsafe {
            znx::znx_zero_i64_ref((n * size) as u64, carry_i64.as_mut_ptr());
        }

        let log_base2k: usize = log_base2k;

        (cols_steps..cols).for_each(|i| {
            izip!(carry_i64.iter_mut(), a.at_mut(i).iter_mut()).for_each(|(ci, xi)| {
                *xi += *ci << log_base2k;
                *ci = get_base_k_carry(*xi, k_rem);
                *xi = (*xi - *ci) >> k_rem;
            });
        })
    }
}

#[inline(always)]
fn get_base_k_carry(x: i64, k: usize) -> i64 {
    (x << 64 - k) >> (64 - k)
}

pub trait VecZnxOps {
    /// Allocates a new [VecZnx].
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of cols.
    fn new_vec_znx(&self, size: usize, cols: usize) -> VecZnx;

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnx] through [VecZnx::from_bytes].
    fn bytes_of_vec_znx(&self, size: usize, cols: usize) -> usize;

    fn vec_znx_normalize_tmp_bytes(&self, size: usize) -> usize;

    /// c <- a + b.
    fn vec_znx_add(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx);

    /// b <- b + a.
    fn vec_znx_add_inplace(&self, b: &mut VecZnx, a: &VecZnx);

    /// c <- a - b.
    fn vec_znx_sub(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx);

    /// b <- a - b.
    fn vec_znx_sub_ab_inplace(&self, b: &mut VecZnx, a: &VecZnx);

    /// b <- b - a.
    fn vec_znx_sub_ba_inplace(&self, b: &mut VecZnx, a: &VecZnx);

    /// b <- -a.
    fn vec_znx_negate(&self, b: &mut VecZnx, a: &VecZnx);

    /// b <- -b.
    fn vec_znx_negate_inplace(&self, a: &mut VecZnx);

    /// b <- a * X^k (mod X^{n} + 1)
    fn vec_znx_rotate(&self, k: i64, b: &mut VecZnx, a: &VecZnx);

    /// a <- a * X^k (mod X^{n} + 1)
    fn vec_znx_rotate_inplace(&self, k: i64, a: &mut VecZnx);

    /// b <- phi_k(a) where phi_k: X^i -> X^{i*k} (mod (X^{n} + 1))
    fn vec_znx_automorphism(&self, k: i64, b: &mut VecZnx, a: &VecZnx);

    /// a <- phi_k(a) where phi_k: X^i -> X^{i*k} (mod (X^{n} + 1))
    fn vec_znx_automorphism_inplace(&self, k: i64, a: &mut VecZnx);

    /// Splits b into subrings and copies them them into a.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of b have the same ring degree
    /// and that b.n() * b.len() <= a.n()
    fn vec_znx_split(&self, b: &mut Vec<VecZnx>, a: &VecZnx, buf: &mut VecZnx);

    /// Merges the subrings a into b.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of a have the same ring degree
    /// and that a.n() * a.len() <= b.n()
    fn vec_znx_merge(&self, b: &mut VecZnx, a: &Vec<VecZnx>);
}

impl VecZnxOps for Module {
    fn new_vec_znx(&self, size: usize, cols: usize) -> VecZnx {
        VecZnx::new(self.n(), size, cols)
    }

    fn bytes_of_vec_znx(&self, size: usize, cols: usize) -> usize {
        bytes_of_vec_znx(self.n(), size, cols)
    }

    fn vec_znx_normalize_tmp_bytes(&self, size: usize) -> usize {
        unsafe { vec_znx::vec_znx_normalize_base2k_tmp_bytes(self.ptr) as usize * size }
    }

    // c <- a + b
    fn vec_znx_add(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx) {
        let n: usize = self.n();
        #[cfg(debug_assertions)]
        {
            assert_eq!(c.n(), n);
            assert_eq!(a.n(), n);
            assert_eq!(b.n(), n);
        }
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
                c.as_mut_ptr(),
                c.cols() as u64,
                (n * c.size()) as u64,
                a.as_ptr(),
                a.cols() as u64,
                (n * a.size()) as u64,
                b.as_ptr(),
                b.cols() as u64,
                (n * b.size()) as u64,
            )
        }
    }

    // b <- a + b
    fn vec_znx_add_inplace(&self, b: &mut VecZnx, a: &VecZnx) {
        let n: usize = self.n();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), n);
            assert_eq!(b.n(), n);
        }
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
                b.as_mut_ptr(),
                b.cols() as u64,
                (n * b.size()) as u64,
                a.as_ptr(),
                a.cols() as u64,
                (n * a.size()) as u64,
                b.as_ptr(),
                b.cols() as u64,
                (n * b.size()) as u64,
            )
        }
    }

    // c <- a + b
    fn vec_znx_sub(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx) {
        let n: usize = self.n();
        #[cfg(debug_assertions)]
        {
            assert_eq!(c.n(), n);
            assert_eq!(a.n(), n);
            assert_eq!(b.n(), n);
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                c.as_mut_ptr(),
                c.cols() as u64,
                (n * c.size()) as u64,
                a.as_ptr(),
                a.cols() as u64,
                (n * a.size()) as u64,
                b.as_ptr(),
                b.cols() as u64,
                (n * b.size()) as u64,
            )
        }
    }

    // b <- a - b
    fn vec_znx_sub_ab_inplace(&self, b: &mut VecZnx, a: &VecZnx) {
        let n: usize = self.n();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), n);
            assert_eq!(b.n(), n);
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                b.as_mut_ptr(),
                b.cols() as u64,
                (n * b.size()) as u64,
                a.as_ptr(),
                a.cols() as u64,
                (n * a.size()) as u64,
                b.as_ptr(),
                b.cols() as u64,
                (n * b.size()) as u64,
            )
        }
    }

    // b <- b - a
    fn vec_znx_sub_ba_inplace(&self, b: &mut VecZnx, a: &VecZnx) {
        let n: usize = self.n();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), n);
            assert_eq!(b.n(), n);
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                b.as_mut_ptr(),
                b.cols() as u64,
                (n * b.size()) as u64,
                b.as_ptr(),
                b.cols() as u64,
                (n * b.size()) as u64,
                a.as_ptr(),
                a.cols() as u64,
                (n * a.size()) as u64,
            )
        }
    }

    fn vec_znx_negate(&self, b: &mut VecZnx, a: &VecZnx) {
        let n: usize = self.n();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), n);
            assert_eq!(b.n(), n);
        }
        unsafe {
            vec_znx::vec_znx_negate(
                self.ptr,
                b.as_mut_ptr(),
                b.cols() as u64,
                (n * b.size()) as u64,
                a.as_ptr(),
                a.cols() as u64,
                (n * a.size()) as u64,
            )
        }
    }

    fn vec_znx_negate_inplace(&self, a: &mut VecZnx) {
        let n: usize = self.n();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), n);
        }
        unsafe {
            vec_znx::vec_znx_negate(
                self.ptr,
                a.as_mut_ptr(),
                a.cols() as u64,
                (n * a.size()) as u64,
                a.as_ptr(),
                a.cols() as u64,
                (n * a.size()) as u64,
            )
        }
    }

    fn vec_znx_rotate(&self, k: i64, b: &mut VecZnx, a: &VecZnx) {
        let n: usize = self.n();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), n);
            assert_eq!(b.n(), n);
        }
        unsafe {
            vec_znx::vec_znx_rotate(
                self.ptr,
                k,
                b.as_mut_ptr(),
                b.cols() as u64,
                (n * b.size()) as u64,
                a.as_ptr(),
                a.cols() as u64,
                (n * a.size()) as u64,
            )
        }
    }

    fn vec_znx_rotate_inplace(&self, k: i64, a: &mut VecZnx) {
        let n: usize = self.n();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), n);
        }
        unsafe {
            vec_znx::vec_znx_rotate(
                self.ptr,
                k,
                a.as_mut_ptr(),
                a.cols() as u64,
                (n * a.size()) as u64,
                a.as_ptr(),
                a.cols() as u64,
                (n * a.size()) as u64,
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
    fn vec_znx_automorphism(&self, k: i64, b: &mut VecZnx, a: &VecZnx) {
        let n: usize = self.n();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), n);
            assert_eq!(b.n(), n);
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.ptr,
                k,
                b.as_mut_ptr(),
                b.cols() as u64,
                (n * b.size()) as u64,
                a.as_ptr(),
                a.cols() as u64,
                (n * a.size()) as u64,
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
    fn vec_znx_automorphism_inplace(&self, k: i64, a: &mut VecZnx) {
        let n: usize = self.n();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), n);
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.ptr,
                k,
                a.as_mut_ptr(),
                a.cols() as u64,
                (n * a.size()) as u64,
                a.as_ptr(),
                a.cols() as u64,
                (n * a.size()) as u64,
            );
        }
    }

    fn vec_znx_split(&self, b: &mut Vec<VecZnx>, a: &VecZnx, buf: &mut VecZnx) {
        let (n_in, n_out) = (a.n(), b[0].n());

        debug_assert!(
            n_out < n_in,
            "invalid a: output ring degree should be smaller"
        );
        b[1..].iter().for_each(|bi| {
            debug_assert_eq!(
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

    fn vec_znx_merge(&self, b: &mut VecZnx, a: &Vec<VecZnx>) {
        let (n_in, n_out) = (b.n(), a[0].n());

        debug_assert!(
            n_out < n_in,
            "invalid a: output ring degree should be smaller"
        );
        a[1..].iter().for_each(|ai| {
            debug_assert_eq!(
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
