use crate::Backend;
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

    /// The number of polynomials
    pub cols: usize,

    /// The number of limbs per polynomial (a.k.a small polynomials).
    pub limbs: usize,

    /// Polynomial coefficients, as a contiguous array. Each col is equally spaced by n.
    pub data: Vec<i64>,

    /// Pointer to data (data can be enpty if [VecZnx] borrows space instead of owning it).
    pub ptr: *mut i64,
}

pub fn bytes_of_vec_znx(n: usize, cols: usize, limbs: usize) -> usize {
    n * cols * limbs * size_of::<i64>()
}

impl VecZnx {
    /// Returns a new struct implementing [VecZnx] with the provided data as backing array.
    ///
    /// The struct will take ownership of buf[..[Self::bytes_of]]
    ///
    /// User must ensure that data is properly alligned and that
    /// the limbs of data is equal to [Self::bytes_of].
    pub fn from_bytes(n: usize, cols: usize, limbs: usize, bytes: &mut [u8]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(cols > 0);
            assert!(limbs > 0);
            assert_eq!(bytes.len(), Self::bytes_of(n, cols, limbs));
            assert_alignement(bytes.as_ptr());
        }
        unsafe {
            let bytes_i64: &mut [i64] = cast_mut::<u8, i64>(bytes);
            let ptr: *mut i64 = bytes_i64.as_mut_ptr();
            Self {
                n: n,
                cols: cols,
                limbs: limbs,
                data: Vec::from_raw_parts(ptr, bytes.len(), bytes.len()),
                ptr: ptr,
            }
        }
    }

    pub fn from_bytes_borrow(n: usize, cols: usize, limbs: usize, bytes: &mut [u8]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(cols > 0);
            assert!(limbs > 0);
            assert!(bytes.len() >= Self::bytes_of(n, cols, limbs));
            assert_alignement(bytes.as_ptr());
        }
        Self {
            n: n,
            cols: cols,
            limbs: limbs,
            data: Vec::new(),
            ptr: bytes.as_mut_ptr() as *mut i64,
        }
    }

    pub fn bytes_of(n: usize, cols: usize, limbs: usize) -> usize {
        bytes_of_vec_znx(n, cols, limbs)
    }

    pub fn copy_from(&mut self, a: &Self) {
        copy_vec_znx_from(self, a);
    }

    pub fn borrowing(&self) -> bool {
        self.data.len() == 0
    }

    /// Total limbs is [Self::n()] * [Self::poly_count()].
    pub fn raw(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.n * self.poly_count()) }
    }

    /// Returns a reference to backend slice of the receiver.
    /// Total size is [Self::n()] * [Self::poly_count()].
    pub fn raw_mut(&mut self) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.n * self.poly_count()) }
    }

    /// Returns a non-mutable pointer to the backedn slice of the receiver.
    pub fn as_ptr(&self) -> *const i64 {
        self.ptr
    }

    /// Returns a mutable pointer to the backedn slice of the receiver.
    pub fn as_mut_ptr(&mut self) -> *mut i64 {
        self.ptr
    }

    /// Returns a non-mutable pointer starting a the (i, j)-th small poly.
    pub fn at_ptr(&self, i: usize, j: usize) -> *const i64 {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.cols());
            assert!(j < self.limbs());
        }
        let offset: usize = self.n * (j * self.cols() + i);
        self.ptr.wrapping_add(offset)
    }

    /// Returns a non-mutable reference to the i-th limb.
    /// The returned array is of size [Self::n()] * [Self::cols()].
    pub fn at_limb(&self, i: usize) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.at_ptr(0, i), self.n * self.cols()) }
    }

    /// Returns a non-mutable reference to the (i, j)-th poly.
    /// The returned array is of size [Self::n()].
    pub fn at_poly(&self, i: usize, j: usize) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.at_ptr(i, j), self.n) }
    }

    /// Returns a mutable pointer starting a the (i, j)-th small poly.
    pub fn at_mut_ptr(&mut self, i: usize, j: usize) -> *mut i64 {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.cols());
            assert!(j < self.limbs());
        }

        let offset: usize = self.n * (j * self.cols() + i);
        self.ptr.wrapping_add(offset)
    }

    /// Returns a mutable reference to the i-th limb.
    /// The returned array is of size [Self::n()] * [Self::cols()].
    pub fn at_limb_mut(&mut self, i: usize) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.at_mut_ptr(0, i), self.n * self.cols()) }
    }

    /// Returns a mutable reference to the (i, j)-th poly.
    /// The returned array is of size [Self::n()].
    pub fn at_poly_mut(&mut self, i: usize, j: usize) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.at_mut_ptr(i, j), self.n) }
    }

    pub fn zero(&mut self) {
        unsafe { znx::znx_zero_i64_ref((self.n * self.poly_count()) as u64, self.ptr) }
    }

    pub fn normalize(&mut self, log_base2k: usize, carry: &mut [u8]) {
        normalize(log_base2k, self, carry)
    }

    pub fn rsh(&mut self, log_base2k: usize, k: usize, carry: &mut [u8]) {
        rsh(log_base2k, self, k, carry)
    }

    pub fn switch_degree(&self, a: &mut Self) {
        switch_degree(a, self)
    }

    // Prints the first `n` coefficients of each limb
    pub fn print(&self, n: usize) {
        (0..self.limbs()).for_each(|i| println!("{}: {:?}", i, &self.at_limb(i)[..n]))
    }
}

impl Infos for VecZnx {
    fn n(&self) -> usize {
        self.n
    }

    fn log_n(&self) -> usize {
        (usize::BITS - (self.n() - 1).leading_zeros()) as _
    }

    fn rows(&self) -> usize {
        1
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn limbs(&self) -> usize {
        self.limbs
    }

    fn poly_count(&self) -> usize {
        self.cols * self.limbs
    }
}

/// Copies the coefficients of `a` on the receiver.
/// Copy is done with the minimum size matching both backing arrays.
/// Panics if the cols do not match.
pub fn copy_vec_znx_from(b: &mut VecZnx, a: &VecZnx) {
    assert_eq!(b.cols(), a.cols());
    let data_a: &[i64] = a.raw();
    let data_b: &mut [i64] = b.raw_mut();
    let size = min(data_b.len(), data_a.len());
    data_b[..size].copy_from_slice(&data_a[..size])
}

impl VecZnx {
    /// Allocates a new [VecZnx] composed of #size polynomials of Z\[X\].
    pub fn new(n: usize, cols: usize, limbs: usize) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(n > 0);
            assert!(n & (n - 1) == 0);
            assert!(cols > 0);
            assert!(limbs > 0);
        }
        let mut data: Vec<i64> = alloc_aligned::<i64>(n * cols * limbs);
        let ptr: *mut i64 = data.as_mut_ptr();
        Self {
            n: n,
            cols: cols,
            limbs: limbs,
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
                .truncate(self.n() * self.cols() * (self.limbs() - k / log_base2k));
        }

        self.limbs -= k / log_base2k;

        let k_rem: usize = k % log_base2k;

        if k_rem != 0 {
            let mask: i64 = ((1 << (log_base2k - k_rem - 1)) - 1) << k_rem;
            self.at_limb_mut(self.limbs() - 1)
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

    let limbs: usize = min(a.limbs(), b.limbs());

    (0..limbs).for_each(|i| {
        izip!(
            a.at_limb(i).iter().step_by(gap_in),
            b.at_limb_mut(i).iter_mut().step_by(gap_out)
        )
        .for_each(|(x_in, x_out)| *x_out = *x_in);
    });
}

fn normalize_tmp_bytes(n: usize, limbs: usize) -> usize {
    n * limbs * std::mem::size_of::<i64>()
}

fn normalize(log_base2k: usize, a: &mut VecZnx, tmp_bytes: &mut [u8]) {
    let n: usize = a.n();
    let cols: usize = a.cols();

    debug_assert!(
        tmp_bytes.len() >= normalize_tmp_bytes(n, cols),
        "invalid tmp_bytes: tmp_bytes.len()={} < normalize_tmp_bytes({}, {})",
        tmp_bytes.len(),
        n,
        cols,
    );
    #[cfg(debug_assertions)]
    {
        assert_alignement(tmp_bytes.as_ptr())
    }

    let carry_i64: &mut [i64] = cast_mut(tmp_bytes);

    unsafe {
        znx::znx_zero_i64_ref(n as u64, carry_i64.as_mut_ptr());
        (0..a.limbs()).rev().for_each(|i| {
            znx::znx_normalize(
                (n * cols) as u64,
                log_base2k as u64,
                a.at_mut_ptr(0, i),
                carry_i64.as_mut_ptr(),
                a.at_mut_ptr(0, i),
                carry_i64.as_mut_ptr(),
            )
        });
    }
}

pub fn rsh_tmp_bytes(n: usize, limbs: usize) -> usize {
    n * limbs * std::mem::size_of::<i64>()
}

pub fn rsh(log_base2k: usize, a: &mut VecZnx, k: usize, tmp_bytes: &mut [u8]) {
    let n: usize = a.n();
    let limbs: usize = a.limbs();

    #[cfg(debug_assertions)]
    {
        assert!(
            tmp_bytes.len() >= rsh_tmp_bytes(n, limbs),
            "invalid carry: carry.len()/8={} < rsh_tmp_bytes({}, {})",
            tmp_bytes.len() >> 3,
            n,
            limbs,
        );
        assert_alignement(tmp_bytes.as_ptr());
    }

    let limbs: usize = a.limbs();
    let size_steps: usize = k / log_base2k;

    a.raw_mut().rotate_right(n * limbs * size_steps);
    unsafe {
        znx::znx_zero_i64_ref((n * limbs * size_steps) as u64, a.as_mut_ptr());
    }

    let k_rem = k % log_base2k;

    if k_rem != 0 {
        let carry_i64: &mut [i64] = cast_mut(tmp_bytes);

        unsafe {
            znx::znx_zero_i64_ref((n * limbs) as u64, carry_i64.as_mut_ptr());
        }

        let log_base2k: usize = log_base2k;

        (size_steps..limbs).for_each(|i| {
            izip!(carry_i64.iter_mut(), a.at_limb_mut(i).iter_mut()).for_each(|(ci, xi)| {
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
    /// * `cols`: the number of polynomials.
    /// * `limbs`: the number of limbs per polynomial (a.k.a small polynomials).
    fn new_vec_znx(&self, cols: usize, limbs: usize) -> VecZnx;

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnx] through [VecZnx::from_bytes].
    fn bytes_of_vec_znx(&self, cols: usize, size: usize) -> usize;

    fn vec_znx_normalize_tmp_bytes(&self, cols: usize) -> usize;

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

impl<B: Backend> VecZnxOps for Module<B> {
    fn new_vec_znx(&self, cols: usize, limbs: usize) -> VecZnx {
        VecZnx::new(self.n(), cols, limbs)
    }

    fn bytes_of_vec_znx(&self, cols: usize, limbs: usize) -> usize {
        bytes_of_vec_znx(self.n(), cols, limbs)
    }

    fn vec_znx_normalize_tmp_bytes(&self, cols: usize) -> usize {
        unsafe { vec_znx::vec_znx_normalize_base2k_tmp_bytes(self.ptr) as usize * cols }
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
                c.limbs() as u64,
                (n * c.cols()) as u64,
                a.as_ptr(),
                a.limbs() as u64,
                (n * a.cols()) as u64,
                b.as_ptr(),
                b.limbs() as u64,
                (n * b.cols()) as u64,
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
                b.limbs() as u64,
                (n * b.cols()) as u64,
                a.as_ptr(),
                a.limbs() as u64,
                (n * a.cols()) as u64,
                b.as_ptr(),
                b.limbs() as u64,
                (n * b.cols()) as u64,
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
                c.limbs() as u64,
                (n * c.cols()) as u64,
                a.as_ptr(),
                a.limbs() as u64,
                (n * a.cols()) as u64,
                b.as_ptr(),
                b.limbs() as u64,
                (n * b.cols()) as u64,
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
                b.limbs() as u64,
                (n * b.cols()) as u64,
                a.as_ptr(),
                a.limbs() as u64,
                (n * a.cols()) as u64,
                b.as_ptr(),
                b.limbs() as u64,
                (n * b.cols()) as u64,
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
                b.limbs() as u64,
                (n * b.cols()) as u64,
                b.as_ptr(),
                b.limbs() as u64,
                (n * b.cols()) as u64,
                a.as_ptr(),
                a.limbs() as u64,
                (n * a.cols()) as u64,
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
                b.limbs() as u64,
                (n * b.cols()) as u64,
                a.as_ptr(),
                a.limbs() as u64,
                (n * a.cols()) as u64,
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
                a.limbs() as u64,
                (n * a.cols()) as u64,
                a.as_ptr(),
                a.limbs() as u64,
                (n * a.cols()) as u64,
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
                b.limbs() as u64,
                (n * b.cols()) as u64,
                a.as_ptr(),
                a.limbs() as u64,
                (n * a.cols()) as u64,
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
                a.limbs() as u64,
                (n * a.cols()) as u64,
                a.as_ptr(),
                a.limbs() as u64,
                (n * a.cols()) as u64,
            )
        }
    }

    /// Maps X^i to X^{ik} mod X^{n}+1. The mapping is applied independently on each size.
    ///
    /// # Arguments
    ///
    /// * `a`: input.
    /// * `b`: output.
    /// * `k`: the power to which to map each coefficients.
    /// * `a_size`: the number of a_size on which to apply the mapping.
    ///
    /// # Panics
    ///
    /// The method will panic if the argument `a` is greater than `a.limbs()`.
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
                b.limbs() as u64,
                (n * b.cols()) as u64,
                a.as_ptr(),
                a.limbs() as u64,
                (n * a.cols()) as u64,
            );
        }
    }

    /// Maps X^i to X^{ik} mod X^{n}+1. The mapping is applied independently on each size.
    ///
    /// # Arguments
    ///
    /// * `a`: input and output.
    /// * `k`: the power to which to map each coefficients.
    /// * `a_size`: the number of size on which to apply the mapping.
    ///
    /// # Panics
    ///
    /// The method will panic if the argument `size` is greater than `self.limbs()`.
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
                a.limbs() as u64,
                (n * a.cols()) as u64,
                a.as_ptr(),
                a.limbs() as u64,
                (n * a.cols()) as u64,
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
