use crate::cast_mut;
use crate::ffi::vec_znx;
use crate::ffi::znx;
use crate::{alloc_aligned, assert_alignement};
use crate::{Infos, Module};
use itertools::izip;
use std::cmp::min;

/// [VecZnx] represents a vector of small norm polynomials of Zn\[X\] with [i64] coefficients.
/// A [VecZnx] is composed of multiple Zn\[X\] polynomials stored in a single contiguous array
/// in the memory.
#[derive(Clone)]
pub struct VecZnx {
    /// Polynomial degree.
    n: usize,

    /// Number of columns.
    cols: usize,

    /// Polynomial coefficients, as a contiguous array. Each col is equally spaced by n.
    data: Vec<i64>,

    /// Pointer to data (data can be enpty if [VecZnx] borrows space instead of owning it).
    ptr: *mut i64,
}

pub trait VecZnxVec {
    fn dblptr(&self) -> Vec<&[i64]>;
    fn dblptr_mut(&mut self) -> Vec<&mut [i64]>;
}

impl VecZnxVec for Vec<VecZnx> {
    fn dblptr(&self) -> Vec<&[i64]> {
        self.iter().map(|v| v.raw()).collect()
    }

    fn dblptr_mut(&mut self) -> Vec<&mut [i64]> {
        self.iter_mut().map(|v| v.raw_mut()).collect()
    }
}

pub fn bytes_of_vec_znx(n: usize, cols: usize) -> usize {
    n * cols * 8
}

impl VecZnx {
    /// Returns a new struct implementing [VecZnx] with the provided data as backing array.
    ///
    /// The struct will take ownership of buf[..[VecZnx::bytes_of]]
    ///
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [VecZnx::bytes_of].
    pub fn from_bytes(n: usize, cols: usize, bytes: &mut [u8]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert_eq!(bytes.len(), Self::bytes_of(n, cols));
            assert_alignement(bytes.as_ptr());
        }
        unsafe {
            let bytes_i64: &mut [i64] = cast_mut::<u8, i64>(bytes);
            let ptr: *mut i64 = bytes_i64.as_mut_ptr();
            VecZnx {
                n: n,
                cols: cols,
                data: Vec::from_raw_parts(bytes_i64.as_mut_ptr(), bytes.len(), bytes.len()),
                ptr: ptr,
            }
        }
    }

    pub fn from_bytes_borrow(n: usize, cols: usize, bytes: &mut [u8]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(bytes.len() >= Self::bytes_of(n, cols));
            assert_alignement(bytes.as_ptr());
        }
        VecZnx {
            n: n,
            cols: cols,
            data: Vec::new(),
            ptr: bytes.as_mut_ptr() as *mut i64,
        }
    }

    pub fn bytes_of(n: usize, cols: usize) -> usize {
        bytes_of_vec_znx(n, cols)
    }

    pub fn copy_from(&mut self, a: &VecZnx) {
        copy_vec_znx_from(self, a);
    }

    pub fn raw(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.n * self.cols) }
    }

    pub fn borrowing(&self) -> bool {
        self.data.len() == 0
    }

    pub fn raw_mut(&mut self) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.n * self.cols) }
    }

    pub fn as_ptr(&self) -> *const i64 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut i64 {
        self.ptr
    }

    pub fn at(&self, i: usize) -> &[i64] {
        let n: usize = self.n();
        &self.raw()[n * i..n * (i + 1)]
    }

    pub fn at_mut(&mut self, i: usize) -> &mut [i64] {
        let n: usize = self.n();
        &mut self.raw_mut()[n * i..n * (i + 1)]
    }

    pub fn at_ptr(&self, i: usize) -> *const i64 {
        self.ptr.wrapping_add(i * self.n)
    }

    pub fn at_mut_ptr(&mut self, i: usize) -> *mut i64 {
        self.ptr.wrapping_add(i * self.n)
    }

    pub fn zero(&mut self) {
        unsafe { znx::znx_zero_i64_ref((self.n * self.cols) as u64, self.ptr) }
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

    pub fn print(&self, cols: usize, n: usize) {
        (0..cols).for_each(|i| println!("{}: {:?}", i, &self.at(i)[..n]))
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
    pub fn new(n: usize, cols: usize) -> Self {
        let mut data: Vec<i64> = alloc_aligned::<i64>(n * cols);
        let ptr: *mut i64 = data.as_mut_ptr();
        Self {
            n: n,
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
                .truncate((self.cols() - k / log_base2k) * self.n());
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

fn normalize(log_base2k: usize, a: &mut VecZnx, tmp_bytes: &mut [u8]) {
    let n: usize = a.n();

    debug_assert!(
        tmp_bytes.len() >= n * 8,
        "invalid tmp_bytes: tmp_bytes.len()={} < self.n()={}",
        tmp_bytes.len(),
        n
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

pub fn rsh(log_base2k: usize, a: &mut VecZnx, k: usize, tmp_bytes: &mut [u8]) {
    let n: usize = a.n();

    debug_assert!(
        tmp_bytes.len() >> 3 >= n,
        "invalid carry: carry.len()/8={} < self.n()={}",
        tmp_bytes.len() >> 3,
        n
    );

    #[cfg(debug_assertions)]
    {
        assert_alignement(tmp_bytes.as_ptr())
    }

    let cols: usize = a.cols();
    let cols_steps: usize = k / log_base2k;

    a.raw_mut().rotate_right(n * cols_steps);
    unsafe {
        znx::znx_zero_i64_ref((n * cols_steps) as u64, a.as_mut_ptr());
    }

    let k_rem = k % log_base2k;

    if k_rem != 0 {
        let carry_i64: &mut [i64] = cast_mut(tmp_bytes);

        unsafe {
            znx::znx_zero_i64_ref(n as u64, carry_i64.as_mut_ptr());
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
    fn new_vec_znx(&self, cols: usize) -> VecZnx;

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnx] through [VecZnx::from_bytes].
    fn bytes_of_vec_znx(&self, cols: usize) -> usize;

    fn vec_znx_normalize_tmp_bytes(&self) -> usize;

    /// c <- a + b.
    fn vec_znx_add(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx);

    /// b <- b + a.
    fn vec_znx_add_inplace(&self, b: &mut VecZnx, a: &VecZnx);

    /// c <- a - b.
    fn vec_znx_sub(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx);

    /// b <- b - a.
    fn vec_znx_sub_inplace(&self, b: &mut VecZnx, a: &VecZnx);

    /// b <- -a.
    fn vec_znx_negate(&self, b: &mut VecZnx, a: &VecZnx);

    /// b <- -b.
    fn vec_znx_negate_inplace(&self, a: &mut VecZnx);

    /// b <- a * X^k (mod X^{n} + 1)
    fn vec_znx_rotate(&self, k: i64, b: &mut VecZnx, a: &VecZnx);

    /// a <- a * X^k (mod X^{n} + 1)
    fn vec_znx_rotate_inplace(&self, k: i64, a: &mut VecZnx);

    /// b <- phi_k(a) where phi_k: X^i -> X^{i*k} (mod (X^{n} + 1))
    fn vec_znx_automorphism(&self, k: i64, b: &mut VecZnx, a: &VecZnx, a_cols: usize);

    /// a <- phi_k(a) where phi_k: X^i -> X^{i*k} (mod (X^{n} + 1))
    fn vec_znx_automorphism_inplace(&self, k: i64, a: &mut VecZnx, a_cols: usize);

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
    fn new_vec_znx(&self, cols: usize) -> VecZnx {
        VecZnx::new(self.n(), cols)
    }

    fn bytes_of_vec_znx(&self, cols: usize) -> usize {
        self.n() * cols * 8
    }

    fn vec_znx_normalize_tmp_bytes(&self) -> usize {
        unsafe { vec_znx::vec_znx_normalize_base2k_tmp_bytes(self.ptr) as usize }
    }

    // c <- a + b
    fn vec_znx_add(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx) {
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
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
    fn vec_znx_add_inplace(&self, b: &mut VecZnx, a: &VecZnx) {
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
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
    fn vec_znx_sub(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx) {
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
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
    fn vec_znx_sub_inplace(&self, b: &mut VecZnx, a: &VecZnx) {
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
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

    fn vec_znx_negate(&self, b: &mut VecZnx, a: &VecZnx) {
        unsafe {
            vec_znx::vec_znx_negate(
                self.ptr,
                b.as_mut_ptr(),
                b.cols() as u64,
                b.n() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
            )
        }
    }

    fn vec_znx_negate_inplace(&self, a: &mut VecZnx) {
        unsafe {
            vec_znx::vec_znx_negate(
                self.ptr,
                a.as_mut_ptr(),
                a.cols() as u64,
                a.n() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
            )
        }
    }

    fn vec_znx_rotate(&self, k: i64, b: &mut VecZnx, a: &VecZnx) {
        unsafe {
            vec_znx::vec_znx_rotate(
                self.ptr,
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

    fn vec_znx_rotate_inplace(&self, k: i64, a: &mut VecZnx) {
        unsafe {
            vec_znx::vec_znx_rotate(
                self.ptr,
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
    /// use base2k::{Module, MODULETYPE, VecZnx, Encoding, Infos, VecZnxOps};
    /// use itertools::izip;
    ///
    /// let n: usize = 8; // polynomial degree
    /// let module = Module::new(n, MODULETYPE::FFT64);
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
    /// izip!(b.raw().iter(), c.raw().iter()).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
    /// ```
    fn vec_znx_automorphism(&self, k: i64, b: &mut VecZnx, a: &VecZnx, a_cols: usize) {
        debug_assert_eq!(a.n(), self.n());
        debug_assert_eq!(b.n(), self.n());
        debug_assert!(a.cols() >= a_cols);
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.ptr,
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
    /// use base2k::{Module, MODULETYPE, VecZnx, Encoding, Infos, VecZnxOps};
    /// use itertools::izip;
    ///
    /// let n: usize = 8; // polynomial degree
    /// let module = Module::new(n, MODULETYPE::FFT64);
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
    /// izip!(a.raw().iter(), b.raw().iter()).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
    /// ```
    fn vec_znx_automorphism_inplace(&self, k: i64, a: &mut VecZnx, a_cols: usize) {
        debug_assert_eq!(a.n(), self.n());
        debug_assert!(a.cols() >= a_cols);
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.ptr,
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
