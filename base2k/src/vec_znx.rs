use crate::cast_mut_u8_to_mut_i64_slice;
use crate::ffi::vec_znx;
use crate::ffi::znx;
use crate::{Infos, Module};
use itertools::izip;
use std::cmp::min;

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
            data: vec![i64::default(); Self::buffer_size(n, limbs)],
        }
    }

    /// Returns the minimum size of the [i64] array required to assign a
    /// new backend array to a [VecZnx] through [VecZnx::from_buffer].
    pub fn buffer_size(n: usize, limbs: usize) -> usize {
        n * limbs
    }

    /// Assigns a new backing array to a [VecZnx].
    pub fn from_buffer(&mut self, n: usize, limbs: usize, buf: &mut [i64]) {
        let size = Self::buffer_size(n, limbs);
        assert!(
            buf.len() >= size,
            "invalid buffer: buf.len()={} < self.buffer_size(n={}, limbs={})={}",
            buf.len(),
            n,
            limbs,
            size
        );
        self.n = n;
        self.data = Vec::from(&buf[..size])
    }

    /// Copies the coefficients of `a` on the receiver.
    /// Copy is done with the minimum size matching both backing arrays.
    pub fn copy_from(&mut self, a: &VecZnx) {
        let size = min(self.data.len(), a.data.len());
        self.data[..size].copy_from_slice(&a.data[..size])
    }

    /// Returns a non-mutable pointer to the backing array of the [VecZnx].
    pub fn as_ptr(&self) -> *const i64 {
        self.data.as_ptr()
    }

    /// Returns a mutable pointer to the backing array of the [VecZnx].
    pub fn as_mut_ptr(&mut self) -> *mut i64 {
        self.data.as_mut_ptr()
    }

    /// Returns a non-mutable reference to the i-th limb of the [VecZnx].
    pub fn at(&self, i: usize) -> &[i64] {
        &self.data[i * self.n..(i + 1) * self.n]
    }

    /// Returns a mutable reference to the i-th limb of the [VecZnx].
    pub fn at_mut(&mut self, i: usize) -> &mut [i64] {
        &mut self.data[i * self.n..(i + 1) * self.n]
    }

    /// Returns a non-mutable pointer to the i-th limb of the [VecZnx].
    pub fn at_ptr(&self, i: usize) -> *const i64 {
        &self.data[i * self.n] as *const i64
    }

    /// Returns a mutable pointer to the i-th limb of the [VecZnx].
    pub fn at_mut_ptr(&mut self, i: usize) -> *mut i64 {
        &mut self.data[i * self.n] as *mut i64
    }

    /// Zeroes the backing array of the [VecZnx].
    pub fn zero(&mut self) {
        unsafe { znx::znx_zero_i64_ref(self.data.len() as u64, self.data.as_mut_ptr()) }
    }

    /// Normalizes the [VecZnx], ensuring all coefficients are in the interval \[-2^log_base2k, 2^log_base2k].
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: the base two logarithm of the base to reduce to.
    /// * `carry`: scratch space of size at least self.n()<<3.
    ///
    /// # Panics
    ///
    /// The method will panic if carry.len() < self.data.len()*8.
    ///
    /// # Example
    /// ```
    /// use base2k::{VecZnx, Encoding, Infos};
    /// use itertools::izip;
    /// use sampling::source::Source;
    ///
    /// let n: usize = 8; // polynomial degree
    /// let log_base2k: usize = 17; // base two logarithm of the coefficients decomposition
    /// let limbs: usize = 5; // number of limbs (i.e. can store coeffs in the range +/- 2^{limbs * log_base2k - 1})
    /// let log_k: usize = limbs * log_base2k - 5;
    /// let mut a: VecZnx = VecZnx::new(n, limbs);
    /// let mut carry: Vec<u8> = vec![u8::default(); a.n()<<3];
    /// let mut have: Vec<i64> = vec![i64::default(); a.n()];
    /// let mut source = Source::new([1; 32]);
    ///
    /// // Populates the first limb of the of polynomials with random i64 values.
    /// have.iter_mut().for_each(|x| {
    ///     *x = source
    ///         .next_u64n(u64::MAX, u64::MAX)
    ///         .wrapping_sub(u64::MAX / 2 + 1) as i64;
    /// });
    /// a.encode_vec_i64(log_base2k, log_k, &have, 63);
    /// a.normalize(log_base2k, &mut carry);
    ///
    /// // Ensures normalized values are in the range +/- 2^{log_base2k-1}
    /// let base_half = 1 << (log_base2k - 1);
    /// a.data
    ///     .iter()
    ///     .for_each(|x| assert!(x.abs() <= base_half, "|x|={} > 2^(k-1)={}", x, base_half));
    ///
    /// // Ensures reconstructed normalized values are equal to non-normalized values.
    /// let mut want = vec![i64::default(); n];
    /// a.decode_vec_i64(log_base2k, log_k, &mut want);
    /// izip!(want, have).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
    /// ```
    pub fn normalize(&mut self, log_base2k: usize, carry: &mut [u8]) {
        assert!(
            carry.len() >= self.n * 8,
            "invalid carry: carry.len()={} < self.n()={}",
            carry.len(),
            self.n()
        );

        let carry_i64: &mut [i64] = cast_mut_u8_to_mut_i64_slice(carry);

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

    /// Maps X^i to X^{ik} mod X^{n}+1. The mapping is applied independently on each limb.
    ///
    /// # Arguments
    ///
    /// * `k`: the power to which to map each coefficients.
    /// * `limbs`: the number of limbs on which to apply the mapping.
    ///
    /// # Panics
    ///
    /// The method will panic if the argument `limbs` is greater than `self.limbs()`.
    ///
    /// # Example
    /// ```
    /// use base2k::{VecZnx, Encoding, Infos};
    /// use itertools::izip;
    ///
    /// let n: usize = 8; // polynomial degree
    /// let mut a: VecZnx = VecZnx::new(n, 2);
    /// let mut b: VecZnx = VecZnx::new(n, 2);
    ///
    /// (0..a.limbs()).for_each(|i|{
    ///     a.at_mut(i).iter_mut().enumerate().for_each(|(i, x)|{
    ///         *x = i as i64
    ///     })
    /// });
    ///
    /// b.copy_from(&a);
    ///
    /// a.automorphism_inplace(-1, 1); // X^i -> X^(-i)
    /// let limb = b.at_mut(0);
    /// (1..limb.len()).for_each(|i|{
    ///     limb[n-i] = -(i as i64)
    /// });
    /// izip!(a.data.iter(), b.data.iter()).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
    /// ```
    pub fn automorphism_inplace(&mut self, k: i64, limbs: usize) {
        assert!(
            limbs <= self.limbs(),
            "invalid limbs argument: limbs={} > self.limbs()={}",
            limbs,
            self.limbs()
        );
        unsafe {
            (0..limbs).for_each(|i| {
                znx::znx_automorphism_inplace_i64(self.n as u64, k, self.at_mut_ptr(i))
            })
        }
    }

    /// Maps X^i to X^{ik} mod X^{n}+1. The mapping is applied independently on each limb.
    ///
    /// # Arguments
    ///
    /// * `a`: the receiver.
    /// * `k`: the power to which to map each coefficients.
    /// * `limbs`: the number of limbs on which to apply the mapping.
    ///
    /// # Panics
    ///
    /// The method will panic if the argument `limbs` is greater than `self.limbs()` or `a.limbs()`.
    ///
    /// # Example
    /// ```
    /// use base2k::{VecZnx, Encoding, Infos};
    /// use itertools::izip;
    ///
    /// let n: usize = 8; // polynomial degree
    /// let mut a: VecZnx = VecZnx::new(n, 2);
    /// let mut b: VecZnx = VecZnx::new(n, 2);
    /// let mut c: VecZnx = VecZnx::new(n, 2);
    ///
    /// (0..a.limbs()).for_each(|i|{
    ///     a.at_mut(i).iter_mut().enumerate().for_each(|(i, x)|{
    ///         *x = i as i64
    ///     })
    /// });
    ///
    /// a.automorphism(&mut b, -1, 1); // X^i -> X^(-i)
    /// let limb = c.at_mut(0);
    /// (1..limb.len()).for_each(|i|{
    ///     limb[n-i] = -(i as i64)
    /// });
    /// izip!(b.data.iter(), c.data.iter()).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
    /// ```
    pub fn automorphism(&mut self, a: &mut VecZnx, k: i64, limbs: usize) {
        assert!(
            limbs <= self.limbs(),
            "invalid limbs argument: limbs={} > self.limbs()={}",
            limbs,
            self.limbs()
        );
        assert!(
            limbs <= a.limbs(),
            "invalid limbs argument: limbs={} > a.limbs()={}",
            limbs,
            a.limbs()
        );
        unsafe {
            (0..limbs).for_each(|i| {
                znx::znx_automorphism_i64(self.n as u64, k, a.at_mut_ptr(i), self.at_ptr(i))
            })
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
        assert!(
            carry.len() >> 3 >= self.n(),
            "invalid carry: carry.len()/8={} < self.n()={}",
            carry.len() >> 3,
            self.n()
        );

        let limbs: usize = self.limbs();
        let limbs_steps: usize = k / log_base2k;

        self.data.rotate_right(self.n * limbs_steps);
        unsafe {
            znx::znx_zero_i64_ref((self.n * limbs_steps) as u64, self.data.as_mut_ptr());
        }

        let k_rem = k % log_base2k;

        if k_rem != 0 {
            let carry_i64: &mut [i64] = cast_mut_u8_to_mut_i64_slice(carry);

            unsafe {
                znx::znx_zero_i64_ref(self.n() as u64, carry_i64.as_mut_ptr());
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
    fn vec_znx_automorphism(&self, k: i64, b: &mut VecZnx, a: &VecZnx);

    /// a <- phi_k(a) where phi_k: X^i -> X^{i*k} (mod (X^{n} + 1))
    fn vec_znx_automorphism_inplace(&self, k: i64, a: &mut VecZnx);
}

impl VecZnxOps for Module {
    fn new_vec_znx(&self, limbs: usize) -> VecZnx {
        VecZnx::new(self.n(), limbs)
    }

    // c <- a + b
    fn vec_znx_add(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx) {
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
    fn vec_znx_add_inplace(&self, b: &mut VecZnx, a: &VecZnx) {
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
    fn vec_znx_sub(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx) {
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
    fn vec_znx_sub_inplace(&self, b: &mut VecZnx, a: &VecZnx) {
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

    fn vec_znx_negate(&self, b: &mut VecZnx, a: &VecZnx) {
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

    fn vec_znx_negate_inplace(&self, a: &mut VecZnx) {
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

    fn vec_znx_rotate(&self, k: i64, a: &mut VecZnx, b: &VecZnx) {
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

    fn vec_znx_rotate_inplace(&self, k: i64, a: &mut VecZnx) {
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

    fn vec_znx_automorphism(&self, k: i64, b: &mut VecZnx, a: &VecZnx) {
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.0,
                k,
                b.as_mut_ptr(),
                b.limbs() as u64,
                b.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
            );
        }
    }

    fn vec_znx_automorphism_inplace(&self, k: i64, a: &mut VecZnx) {
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.0,
                k,
                a.as_mut_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
            );
        }
    }
}
