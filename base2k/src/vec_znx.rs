use crate::cast_mut;
use crate::ffi::vec_znx;
use crate::ffi::znx;
use crate::ffi::znx::znx_zero_i64_ref;
use crate::{alias_mut_slice_to_vec, alloc_aligned};
use crate::{Infos, Module};
use itertools::izip;
use std::cmp::min;

pub trait VecZnxApi {
    /// Returns the minimum size of the [u8] array required to assign a
    /// new backend array to a [VecZnx] through [VecZnx::from_bytes].
    fn bytes_of(n: usize, limbs: usize) -> usize;
    fn as_ptr(&self) -> *const i64;
    fn as_mut_ptr(&mut self) -> *mut i64;
    fn at(&self, i: usize) -> &[i64];
    fn at_mut(&mut self, i: usize) -> &mut [i64];
    fn at_ptr(&self, i: usize) -> *const i64;
    fn at_mut_ptr(&mut self, i: usize) -> *mut i64;
    fn zero(&mut self);
    fn normalize(&mut self, log_base2k: usize, carry: &mut [u8]);
}

pub fn bytes_of_vec_znx(n: usize, limbs: usize) -> usize {
    n * limbs * 8
}

pub struct VecZnxBorrow {
    pub n: usize,
    pub limbs: usize,
    pub data: *mut i64,
}

impl VecZnxBorrow {
    /// Returns a new struct implementing [VecZnxApi] with the provided data as backing array.
    ///
    /// The struct will take ownership of buf[..[VecZnx::bytes_of]]
    ///
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [Module::bytes_of_vec_znx].
    pub fn from_bytes(n: usize, limbs: usize, bytes: &mut [u8]) -> impl VecZnxApi {
        let size = Self::bytes_of(n, limbs);
        assert!(
            bytes.len() >= size,
            "invalid buffer: buf.len()={} < self.buffer_size(n={}, limbs={})={}",
            bytes.len(),
            n,
            limbs,
            size
        );
        VecZnxBorrow {
            n: n,
            limbs: limbs,
            data: cast_mut(&mut bytes[..size]).as_mut_ptr(),
        }
    }
}

impl VecZnxApi for VecZnxBorrow {
    fn bytes_of(n: usize, limbs: usize) -> usize {
        bytes_of_vec_znx(n, limbs)
    }

    fn as_ptr(&self) -> *const i64 {
        self.data
    }

    fn as_mut_ptr(&mut self) -> *mut i64 {
        self.data
    }

    fn at(&self, i: usize) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.data.wrapping_add(self.n * i), self.n) }
    }

    fn at_mut(&mut self, i: usize) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.at_mut_ptr(i), self.n) }
    }

    fn at_ptr(&self, i: usize) -> *const i64 {
        self.data.wrapping_add(self.n * i)
    }

    fn at_mut_ptr(&mut self, i: usize) -> *mut i64 {
        self.data.wrapping_add(self.n * i)
    }

    fn zero(&mut self) {
        unsafe {
            znx_zero_i64_ref((self.n * self.limbs) as u64, self.data);
        }
    }

    fn normalize(&mut self, log_base2k: usize, carry: &mut [u8]) {
        assert!(
            carry.len() >= self.n() * 8,
            "invalid carry: carry.len()={} < self.n()={}",
            carry.len(),
            self.n()
        );

        let carry_i64: &mut [i64] = cast_mut(carry);

        unsafe {
            znx::znx_zero_i64_ref(self.n() as u64, carry_i64.as_mut_ptr());
            (0..self.limbs()).rev().for_each(|i| {
                znx::znx_normalize(
                    self.n as u64,
                    log_base2k as u64,
                    self.at_mut_ptr(i),
                    carry_i64.as_mut_ptr(),
                    self.at_mut_ptr(i),
                    carry_i64.as_mut_ptr(),
                )
            });
        }
    }
}

impl VecZnx {
    /// Returns a new struct implementing [VecZnxApi] with the provided data as backing array.
    ///
    /// The struct will take ownership of buf[..[VecZnx::bytes_of]]
    ///
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [Module::bytes_of_vec_znx].
    pub fn from_bytes(n: usize, limbs: usize, buf: &mut [u8]) -> impl VecZnxApi {
        let size = Self::bytes_of(n, limbs);
        assert!(
            buf.len() >= size,
            "invalid buffer: buf.len()={} < self.buffer_size(n={}, limbs={})={}",
            buf.len(),
            n,
            limbs,
            size
        );

        VecZnx {
            n: n,
            data: alias_mut_slice_to_vec(cast_mut(&mut buf[..size])),
        }
    }
}

impl VecZnxApi for VecZnx {
    fn bytes_of(n: usize, limbs: usize) -> usize {
        bytes_of_vec_znx(n, limbs)
    }

    /// Returns a non-mutable pointer to the backing array of the [VecZnx].
    fn as_ptr(&self) -> *const i64 {
        self.data.as_ptr()
    }

    /// Returns a mutable pointer to the backing array of the [VecZnx].
    fn as_mut_ptr(&mut self) -> *mut i64 {
        self.data.as_mut_ptr()
    }

    /// Returns a non-mutable reference to the i-th limb of the [VecZnx].
    fn at(&self, i: usize) -> &[i64] {
        &self.data[i * self.n..(i + 1) * self.n]
    }

    /// Returns a mutable reference to the i-th limb of the [VecZnx].
    fn at_mut(&mut self, i: usize) -> &mut [i64] {
        &mut self.data[i * self.n..(i + 1) * self.n]
    }

    /// Returns a non-mutable pointer to the i-th limb of the [VecZnx].
    fn at_ptr(&self, i: usize) -> *const i64 {
        &self.data[i * self.n] as *const i64
    }

    /// Returns a mutable pointer to the i-th limb of the [VecZnx].
    fn at_mut_ptr(&mut self, i: usize) -> *mut i64 {
        &mut self.data[i * self.n] as *mut i64
    }

    /// Zeroes the backing array of the [VecZnx].
    fn zero(&mut self) {
        unsafe { znx::znx_zero_i64_ref(self.data.len() as u64, self.data.as_mut_ptr()) }
    }

    fn normalize(&mut self, log_base2k: usize, carry: &mut [u8]) {
        assert!(
            carry.len() >= self.n() * 8,
            "invalid carry: carry.len()={} < self.n()={}",
            carry.len(),
            self.n()
        );

        let carry_i64: &mut [i64] = cast_mut(carry);

        unsafe {
            znx::znx_zero_i64_ref(self.n() as u64, carry_i64.as_mut_ptr());
            (0..self.limbs()).rev().for_each(|i| {
                znx::znx_normalize(
                    self.n as u64,
                    log_base2k as u64,
                    self.at_mut_ptr(i),
                    carry_i64.as_mut_ptr(),
                    self.at_mut_ptr(i),
                    carry_i64.as_mut_ptr(),
                )
            });
        }
    }
}

/// [VecZnx] represents a vector of small norm polynomials of Zn\[X\] with [i64] coefficients.
/// A [VecZnx] is composed of multiple Zn\[X\] polynomials stored in a single contiguous array
/// in the memory.
#[derive(Clone)]
pub struct VecZnx {
    /// Polynomial degree.
    pub n: usize,
    /// Polynomial coefficients, as a contiguous array. Each limb is equally spaced by n.
    pub data: Vec<i64>,
}

impl VecZnx {
    /// Allocates a new [VecZnx] composed of #limbs polynomials of Z\[X\].
    pub fn new(n: usize, limbs: usize) -> Self {
        Self {
            n: n,
            data: alloc_aligned::<i64>(n * limbs, 64),
        }
    }

    /// Copies the coefficients of `a` on the receiver.
    /// Copy is done with the minimum size matching both backing arrays.
    pub fn copy_from(&mut self, a: &VecZnx) {
        let size = min(self.data.len(), a.data.len());
        self.data[..size].copy_from_slice(&a.data[..size])
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
            .truncate((self.limbs() - k / log_base2k) * self.n());

        let k_rem: usize = k % log_base2k;

        if k_rem != 0 {
            let mask: i64 = ((1 << (log_base2k - k_rem - 1)) - 1) << k_rem;
            self.at_mut(self.limbs() - 1)
                .iter_mut()
                .for_each(|x: &mut i64| *x &= mask)
        }
    }

    /// Right shifts the coefficients by k bits.
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: the base two logarithm of the coefficients decomposition.
    /// * `k`: the shift amount.
    /// * `carry`: scratch space of size at least equal to self.n() * self.limbs() << 3.
    ///
    /// # Panics
    ///
    /// The method will panic if carry.len() < self.n() * self.limbs() << 3.
    pub fn rsh(&mut self, log_base2k: usize, k: usize, carry: &mut [u8]) {
        let n: usize = self.n();

        assert!(
            carry.len() >> 3 >= n,
            "invalid carry: carry.len()/8={} < self.n()={}",
            carry.len() >> 3,
            n
        );

        let limbs: usize = self.limbs();
        let limbs_steps: usize = k / log_base2k;

        self.data.rotate_right(self.n * limbs_steps);
        unsafe {
            znx::znx_zero_i64_ref((self.n * limbs_steps) as u64, self.data.as_mut_ptr());
        }

        let k_rem = k % log_base2k;

        if k_rem != 0 {
            let carry_i64: &mut [i64] = cast_mut(carry);

            unsafe {
                znx::znx_zero_i64_ref(n as u64, carry_i64.as_mut_ptr());
            }

            let mask: i64 = (1 << k_rem) - 1;
            let log_base2k: usize = log_base2k;

            (limbs_steps..limbs).for_each(|i| {
                izip!(carry_i64.iter_mut(), self.at_mut(i).iter_mut()).for_each(|(ci, xi)| {
                    *xi += *ci << log_base2k;
                    *ci = *xi & mask;
                    *xi /= 1 << k_rem;
                });
            })
        }
    }

    /// If self.n() > a.n(): Extracts X^{i*self.n()/a.n()} -> X^{i}.
    /// If self.n() < a.n(): Extracts X^{i} -> X^{i*a.n()/self.n()}.
    ///
    /// # Arguments
    ///
    /// * `a`: the receiver polynomial in which the extracted coefficients are stored.
    pub fn switch_degree(&self, a: &mut VecZnx) {
        let (n_in, n_out) = (self.n(), a.n());
        let (gap_in, gap_out): (usize, usize);

        if n_in > n_out {
            (gap_in, gap_out) = (n_in / n_out, 1)
        } else {
            (gap_in, gap_out) = (1, n_out / n_in);
            a.zero();
        }

        let limbs = min(self.limbs(), a.limbs());

        (0..limbs).for_each(|i| {
            izip!(
                self.at(i).iter().step_by(gap_in),
                a.at_mut(i).iter_mut().step_by(gap_out)
            )
            .for_each(|(x_in, x_out)| *x_out = *x_in);
        });
    }

    pub fn print_limbs(&self, limbs: usize, n: usize) {
        (0..limbs).for_each(|i| println!("{}: {:?}", i, &self.at(i)[..n]))
    }
}

pub trait VecZnxOps {
    /// Allocates a new [VecZnx].
    ///
    /// # Arguments
    ///
    /// * `limbs`: the number of limbs.
    fn new_vec_znx(&self, limbs: usize) -> VecZnx;

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnx] through [VecZnx::from_bytes].
    fn bytes_of_vec_znx(&self, limbs: usize) -> usize;

    /// c <- a + b.
    fn vec_znx_add<T: VecZnxApi + Infos>(&self, c: &mut T, a: &T, b: &T);

    /// b <- b + a.
    fn vec_znx_add_inplace<T: VecZnxApi + Infos>(&self, b: &mut T, a: &T);

    /// c <- a - b.
    fn vec_znx_sub<T: VecZnxApi + Infos>(&self, c: &mut T, a: &T, b: &T);

    /// b <- b - a.
    fn vec_znx_sub_inplace<T: VecZnxApi + Infos>(&self, b: &mut T, a: &T);

    /// b <- -a.
    fn vec_znx_negate<T: VecZnxApi + Infos>(&self, b: &mut T, a: &T);

    /// b <- -b.
    fn vec_znx_negate_inplace<T: VecZnxApi + Infos>(&self, a: &mut T);

    /// b <- a * X^k (mod X^{n} + 1)
    fn vec_znx_rotate<T: VecZnxApi + Infos>(&self, k: i64, b: &mut T, a: &T);

    /// a <- a * X^k (mod X^{n} + 1)
    fn vec_znx_rotate_inplace<T: VecZnxApi + Infos>(&self, k: i64, a: &mut T);

    /// b <- phi_k(a) where phi_k: X^i -> X^{i*k} (mod (X^{n} + 1))
    fn vec_znx_automorphism<T: VecZnxApi + Infos>(&self, k: i64, b: &mut T, a: &T, a_limbs: usize);

    /// a <- phi_k(a) where phi_k: X^i -> X^{i*k} (mod (X^{n} + 1))
    fn vec_znx_automorphism_inplace<T: VecZnxApi + Infos>(&self, k: i64, a: &mut T, a_limbs: usize);

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
    fn new_vec_znx(&self, limbs: usize) -> VecZnx {
        VecZnx::new(self.n(), limbs)
    }

    fn bytes_of_vec_znx(&self, limbs: usize) -> usize {
        self.n() * limbs * 8
    }

    // c <- a + b
    fn vec_znx_add<T: VecZnxApi + Infos>(&self, c: &mut T, a: &T, b: &T) {
        unsafe {
            vec_znx::vec_znx_add(
                self.0,
                c.as_mut_ptr(),
                c.limbs() as u64,
                c.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                b.as_ptr(),
                b.limbs() as u64,
                b.n() as u64,
            )
        }
    }

    // b <- a + b
    fn vec_znx_add_inplace<T: VecZnxApi + Infos>(&self, b: &mut T, a: &T) {
        unsafe {
            vec_znx::vec_znx_add(
                self.0,
                b.as_mut_ptr(),
                b.limbs() as u64,
                b.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                b.as_ptr(),
                b.limbs() as u64,
                b.n() as u64,
            )
        }
    }

    // c <- a + b
    fn vec_znx_sub<T: VecZnxApi + Infos>(&self, c: &mut T, a: &T, b: &T) {
        unsafe {
            vec_znx::vec_znx_sub(
                self.0,
                c.as_mut_ptr(),
                c.limbs() as u64,
                c.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                b.as_ptr(),
                b.limbs() as u64,
                b.n() as u64,
            )
        }
    }

    // b <- a + b
    fn vec_znx_sub_inplace<T: VecZnxApi + Infos>(&self, b: &mut T, a: &T) {
        unsafe {
            vec_znx::vec_znx_sub(
                self.0,
                b.as_mut_ptr(),
                b.limbs() as u64,
                b.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                b.as_ptr(),
                b.limbs() as u64,
                b.n() as u64,
            )
        }
    }

    fn vec_znx_negate<T: VecZnxApi + Infos>(&self, b: &mut T, a: &T) {
        unsafe {
            vec_znx::vec_znx_negate(
                self.0,
                b.as_mut_ptr(),
                b.limbs() as u64,
                b.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
            )
        }
    }

    fn vec_znx_negate_inplace<T: VecZnxApi + Infos>(&self, a: &mut T) {
        unsafe {
            vec_znx::vec_znx_negate(
                self.0,
                a.as_mut_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
            )
        }
    }

    fn vec_znx_rotate<T: VecZnxApi + Infos>(&self, k: i64, a: &mut T, b: &T) {
        unsafe {
            vec_znx::vec_znx_rotate(
                self.0,
                k,
                a.as_mut_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                b.as_ptr(),
                b.limbs() as u64,
                b.n() as u64,
            )
        }
    }

    fn vec_znx_rotate_inplace<T: VecZnxApi + Infos>(&self, k: i64, a: &mut T) {
        unsafe {
            vec_znx::vec_znx_rotate(
                self.0,
                k,
                a.as_mut_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
            )
        }
    }

    /// Maps X^i to X^{ik} mod X^{n}+1. The mapping is applied independently on each limbs.
    ///
    /// # Arguments
    ///
    /// * `a`: input.
    /// * `b`: output.
    /// * `k`: the power to which to map each coefficients.
    /// * `limbs_a`: the number of limbs_a on which to apply the mapping.
    ///
    /// # Panics
    ///
    /// The method will panic if the argument `limbs_a` is greater than `a.limbs()`.
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
    /// (0..a.limbs()).for_each(|i|{
    ///     a.at_mut(i).iter_mut().enumerate().for_each(|(i, x)|{
    ///         *x = i as i64
    ///     })
    /// });
    ///
    /// module.vec_znx_automorphism(-1, &mut b, &a, 1); // X^i -> X^(-i)
    /// let limb = c.at_mut(0);
    /// (1..limb.len()).for_each(|i|{
    ///     limb[n-i] = -(i as i64)
    /// });
    /// izip!(b.data.iter(), c.data.iter()).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
    /// ```
    fn vec_znx_automorphism<T: VecZnxApi + Infos>(&self, k: i64, b: &mut T, a: &T, limbs_a: usize) {
        assert_eq!(a.n(), self.n());
        assert_eq!(b.n(), self.n());
        assert!(a.limbs() >= limbs_a);
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.0,
                k,
                b.as_mut_ptr(),
                b.limbs() as u64,
                b.n() as u64,
                a.as_ptr(),
                limbs_a as u64,
                a.n() as u64,
            );
        }
    }

    /// Maps X^i to X^{ik} mod X^{n}+1. The mapping is applied independently on each limbs.
    ///
    /// # Arguments
    ///
    /// * `a`: input and output.
    /// * `k`: the power to which to map each coefficients.
    /// * `limbs_a`: the number of limbs on which to apply the mapping.
    ///
    /// # Panics
    ///
    /// The method will panic if the argument `limbs` is greater than `self.limbs()`.
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
    /// (0..a.limbs()).for_each(|i|{
    ///     a.at_mut(i).iter_mut().enumerate().for_each(|(i, x)|{
    ///         *x = i as i64
    ///     })
    /// });
    ///
    /// module.vec_znx_automorphism_inplace(-1, &mut a, 1); // X^i -> X^(-i)
    /// let limb = b.at_mut(0);
    /// (1..limb.len()).for_each(|i|{
    ///     limb[n-i] = -(i as i64)
    /// });
    /// izip!(a.data.iter(), b.data.iter()).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
    /// ```
    fn vec_znx_automorphism_inplace<T: VecZnxApi + Infos>(
        &self,
        k: i64,
        a: &mut T,
        limbs_a: usize,
    ) {
        assert_eq!(a.n(), self.n());
        assert!(a.limbs() >= limbs_a);
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.0,
                k,
                a.as_mut_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                a.as_ptr(),
                limbs_a as u64,
                a.n() as u64,
            );
        }
    }

    fn vec_znx_split(&self, b: &mut Vec<VecZnx>, a: &VecZnx, buf: &mut VecZnx) {
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
                a.switch_degree(bi);
                self.vec_znx_rotate(-1, buf, a);
            } else {
                buf.switch_degree(bi);
                self.vec_znx_rotate_inplace(-1, buf);
            }
        })
    }

    fn vec_znx_merge(&self, b: &mut VecZnx, a: &Vec<VecZnx>) {
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
            ai.switch_degree(b);
            self.vec_znx_rotate_inplace(-1, b);
        });

        self.vec_znx_rotate_inplace(a.len() as i64, b);
    }
}
