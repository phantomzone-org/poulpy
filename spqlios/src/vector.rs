use crate::bindings::{
    znx_automorphism_i64, znx_automorphism_inplace_i64, znx_normalize, znx_zero_i64_ref,
};
use crate::cast_mut_u8_to_mut_i64_slice;
use itertools::izip;
use rand_distr::{Distribution, Normal};
use sampling::source::Source;
use std::cmp::min;

pub struct Vector {
    pub n: usize,
    pub log_base2k: usize,
    pub prec: usize,
    pub data: Vec<i64>,
}

impl Vector {
    pub fn new(n: usize, log_base2k: usize, prec: usize) -> Self {
        Self {
            n: n,
            log_base2k: log_base2k,
            prec: prec,
            data: vec![i64::default(); Self::buffer_size(n, log_base2k, prec)],
        }
    }

    pub fn buffer_size(n: usize, log_base2k: usize, prec: usize) -> usize {
        n * ((prec + log_base2k - 1) / log_base2k)
    }

    pub fn from_buffer(&mut self, n: usize, log_base2k: usize, prec: usize, buf: &[i64]) {
        let size = Self::buffer_size(n, log_base2k, prec);
        assert!(
            buf.len() >= size,
            "invalid buffer: buf.len()={} < self.buffer_size(n={}, k={}, prec={})={}",
            buf.len(),
            n,
            log_base2k,
            prec,
            size
        );
        self.n = n;
        self.log_base2k = log_base2k;
        self.prec = prec;
        self.data = Vec::from(&buf[..size])
    }

    pub fn log_n(&self) -> u64 {
        (u64::BITS - (self.n - 1).leading_zeros()) as _
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn prec(&self) -> usize {
        self.prec
    }

    pub fn limbs(&self) -> usize {
        self.data.len() / self.n
    }

    pub fn as_ptr(&self) -> *const i64 {
        self.data.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut i64 {
        self.data.as_mut_ptr()
    }

    pub fn at(&self, i: usize) -> &[i64] {
        &self.data[i * self.n..(i + 1) * self.n]
    }

    pub fn at_ptr(&self, i: usize) -> *const i64 {
        &self.data[i * self.n] as *const i64
    }

    pub fn at_mut_ptr(&mut self, i: usize) -> *mut i64 {
        &mut self.data[i * self.n] as *mut i64
    }

    pub fn at_mut(&mut self, i: usize) -> &mut [i64] {
        &mut self.data[i * self.n..(i + 1) * self.n]
    }

    pub fn set_i64(&mut self, data: &[i64], log_max: usize) {
        let size: usize = min(data.len(), self.n());
        let k_rem: usize = self.log_base2k - (self.prec % self.log_base2k);

        // If 2^{log_base2k} * 2^{k_rem} < 2^{63}-1, then we can simply copy
        // values on the last limb.
        // Else we decompose values base2k.
        if log_max + k_rem < 63 || k_rem == self.log_base2k {
            self.at_mut(self.limbs() - 1).copy_from_slice(&data[..size]);
        } else {
            let mask: i64 = (1 << self.log_base2k) - 1;
            let limbs = self.limbs();
            let steps: usize = min(limbs, (log_max + self.log_base2k - 1) / self.log_base2k);
            (limbs - steps..limbs)
                .rev()
                .enumerate()
                .for_each(|(i, i_rev)| {
                    let shift: usize = i * self.log_base2k;
                    izip!(self.at_mut(i_rev)[..size].iter_mut(), data[..size].iter())
                        .for_each(|(y, x)| *y = (x >> shift) & mask);
                })
        }

        // Case where self.prec % self.k != 0.
        if k_rem != self.log_base2k {
            let limbs = self.limbs();
            let steps: usize = min(limbs, (log_max + self.log_base2k - 1) / self.log_base2k);
            (limbs - steps..limbs).rev().for_each(|i| {
                self.at_mut(i)[..size].iter_mut().for_each(|x| *x <<= k_rem);
            })
        }
    }

    pub fn normalize(&mut self, carry: &mut [u8]) {
        assert!(
            carry.len() >= self.n * 8,
            "invalid carry: carry.len()={} < self.n()={}",
            carry.len(),
            self.n()
        );

        let carry_i64: &mut [i64] = cast_mut_u8_to_mut_i64_slice(carry);

        unsafe {
            znx_zero_i64_ref(self.n() as u64, carry_i64.as_mut_ptr());
            (0..self.limbs()).rev().for_each(|i| {
                znx_normalize(
                    self.n as u64,
                    self.log_base2k as u64,
                    self.at_mut_ptr(i),
                    carry_i64.as_mut_ptr(),
                    self.at_mut_ptr(i),
                    carry_i64.as_mut_ptr(),
                )
            });
        }
    }

    pub fn get_i64(&self, data: &mut [i64]) {
        assert!(
            data.len() >= self.n,
            "invalid data: data.len()={} < self.n()={}",
            data.len(),
            self.n
        );
        data.copy_from_slice(self.at(0));
        let rem: usize = self.log_base2k - (self.prec % self.log_base2k);
        (1..self.limbs()).for_each(|i| {
            if i == self.limbs() - 1 && rem != self.log_base2k {
                let k_rem: usize = self.log_base2k - rem;
                izip!(self.at(i).iter(), data.iter_mut()).for_each(|(x, y)| {
                    *y = (*y << k_rem) + (x >> rem);
                });
            } else {
                izip!(self.at(i).iter(), data.iter_mut()).for_each(|(x, y)| {
                    *y = (*y << self.log_base2k) + x;
                });
            }
        })
    }

    pub fn automorphism_inplace(&mut self, gal_el: i64) {
        unsafe {
            (0..self.limbs()).for_each(|i| {
                znx_automorphism_inplace_i64(self.n as u64, gal_el, self.at_mut_ptr(i))
            })
        }
    }
    pub fn automorphism(&mut self, gal_el: i64, a: &mut Vector) {
        unsafe {
            (0..self.limbs()).for_each(|i| {
                znx_automorphism_i64(self.n as u64, gal_el, a.at_mut_ptr(i), self.at_ptr(i))
            })
        }
    }

    pub fn fill_uniform(&mut self, source: &mut Source) {
        let mut base2k: u64 = 1 << self.log_base2k;
        let mut mask: u64 = base2k - 1;
        let mut base2k_half: i64 = (base2k >> 1) as i64;

        let size: usize = self.n() * (self.limbs() - 1);

        self.data[..size]
            .iter_mut()
            .for_each(|x| *x = (source.next_u64n(base2k, mask) as i64) - base2k_half);

        let log_base2k_rem: usize = self.prec % self.log_base2k;

        if log_base2k_rem != 0 {
            base2k = 1 << log_base2k_rem;
            mask = (base2k - 1) << (self.log_base2k - log_base2k_rem);
            base2k_half = ((mask >> 1) + 1) as i64;
        }

        self.data[size..]
            .iter_mut()
            .for_each(|x| *x = (source.next_u64n(base2k, mask) as i64) - base2k_half);
    }

    pub fn add_dist_f64<T: Distribution<f64>>(&mut self, source: &mut Source, dist: T, bound: f64) {
        let log_base2k_rem: usize = self.prec % self.log_base2k;

        if log_base2k_rem != 0 {
            self.at_mut(self.limbs() - 1).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a += (dist_f64.round() as i64) << log_base2k_rem
            });
        } else {
            self.at_mut(self.limbs() - 1).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a += dist_f64.round() as i64
            });
        }
    }

    pub fn add_normal(&mut self, source: &mut Source, sigma: f64, bound: f64) {
        self.add_dist_f64(source, Normal::new(0.0, sigma).unwrap(), bound);
    }

    pub fn trunc_pow2(&mut self, k: usize) {
        if k == 0 {
            return;
        }

        assert!(
            k <= self.prec,
            "invalid argument k: k={} > self.prec()={}",
            k,
            self.prec()
        );

        self.prec -= k;
        self.data
            .truncate((self.limbs() - k / self.log_base2k) * self.n());

        let k_rem: usize = k % self.log_base2k;

        if k_rem != 0 {
            let mask: i64 = ((1 << (self.log_base2k - k_rem - 1)) - 1) << k_rem;
            self.at_mut(self.limbs() - 1)
                .iter_mut()
                .for_each(|x: &mut i64| *x &= mask)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::vector::Vector;
    use itertools::izip;
    use sampling::source::Source;

    #[test]
    fn test_set_get_i64_lo_norm() {
        let n: usize = 32;
        let k: usize = 19;
        let prec: usize = 128;
        let mut a: Vector = Vector::new(n, k, prec);
        let mut have: Vec<i64> = vec![i64::default(); n];
        have.iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = (i as i64) - (n as i64) / 2);
        a.set_i64(&have, 10);
        let mut want = vec![i64::default(); n];
        a.get_i64(&mut want);
        izip!(want, have).for_each(|(a, b)| assert_eq!(a, b));
    }

    #[test]
    fn test_set_get_i64_hi_norm() {
        let n: usize = 8;
        let k: usize = 17;
        let prec: usize = 84;
        let mut a: Vector = Vector::new(n, k, prec);
        let mut have: Vec<i64> = vec![i64::default(); n];
        let mut source = Source::new([1; 32]);
        have.iter_mut().for_each(|x| {
            *x = source
                .next_u64n(u64::MAX, u64::MAX)
                .wrapping_sub(u64::MAX / 2 + 1) as i64;
        });
        a.set_i64(&have, 63);
        //(0..a.limbs()).for_each(|i| println!("i:{} -> {:?}", i, a.at(i)));
        let mut want = vec![i64::default(); n];
        //(0..a.limbs()).for_each(|i| println!("i:{} -> {:?}", i, a.at(i)));
        a.get_i64(&mut want);
        izip!(want, have).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
    }
    #[test]
    fn test_normalize() {
        let n: usize = 8;
        let k: usize = 17;
        let prec: usize = 84;
        let mut a: Vector = Vector::new(n, k, prec);
        let mut have: Vec<i64> = vec![i64::default(); n];
        let mut source = Source::new([1; 32]);
        have.iter_mut().for_each(|x| {
            *x = source
                .next_u64n(u64::MAX, u64::MAX)
                .wrapping_sub(u64::MAX / 2 + 1) as i64;
        });
        a.set_i64(&have, 63);
        let mut carry: Vec<u8> = vec![u8::default(); n * 8];
        a.normalize(&mut carry);

        let base_half = 1 << (k - 1);
        a.data
            .iter()
            .for_each(|x| assert!(x.abs() <= base_half, "|x|={} > 2^(k-1)={}", x, base_half));
        let mut want = vec![i64::default(); n];
        a.get_i64(&mut want);
        izip!(want, have).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
    }
}
