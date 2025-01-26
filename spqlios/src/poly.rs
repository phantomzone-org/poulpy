use crate::{znx_normalize, znx_zero_i64_ref};
use itertools::izip;
use std::cmp::min;

pub struct Poly {
    pub n: usize,
    pub k: usize,
    pub prec: usize,
    pub data: Vec<i64>,
}

impl Poly {
    pub fn new(n: usize, k: usize, prec: usize) -> Self {
        Self {
            n: n,
            k: k,
            prec: prec,
            data: vec![i64::default(); Self::buffer_size(n, k, prec)],
        }
    }

    pub fn buffer_size(n: usize, k: usize, prec: usize) -> usize {
        n * ((prec + k - 1) / k)
    }

    pub fn from_buffer(&mut self, n: usize, k: usize, prec: usize, buf: &[i64]) {
        let size = Self::buffer_size(n, k, prec);
        assert!(
            buf.len() >= size,
            "invalid buffer: buf.len()={} < self.buffer_size(n={}, k={}, prec={})={}",
            buf.len(),
            n,
            k,
            prec,
            size
        );
        self.n = n;
        self.k = k;
        self.prec = prec;
        self.data = Vec::from(&buf[..size])
    }

    pub fn log_n(&self) -> usize {
        (u64::BITS - (self.n - 1).leading_zeros()) as _
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn limbs(&self) -> usize {
        self.data.len() / self.n
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
        let k_rem: usize = self.k - (self.prec % self.k);

        // If 2^{base} * 2^{k_rem} < 2^{63}-1, then we can simply copy
        // values on the last limb.
        // Else we decompose values base k.
        if log_max + k_rem < 63 || k_rem == self.k {
            self.at_mut(self.limbs() - 1).copy_from_slice(&data[..size]);
        } else {
            let mask: i64 = (1 << self.k) - 1;
            let limbs = self.limbs();
            let steps: usize = min(limbs, (log_max + k_rem + self.k - 1) / self.k);
            (limbs - steps..limbs)
                .rev()
                .enumerate()
                .for_each(|(i, i_rev)| {
                    let shift: usize = i * self.k;
                    izip!(self.at_mut(i_rev)[..size].iter_mut(), data[..size].iter())
                        .for_each(|(y, x)| *y = (x >> shift) & mask);
                })
        }

        // Case where self.prec % self.k != 0.
        if k_rem != self.k {
            let limbs = self.limbs();
            let steps: usize = min(limbs, (log_max + k_rem + self.k - 1) / self.k);
            (limbs - steps..limbs).rev().for_each(|i| {
                self.at_mut(i)[..size].iter_mut().for_each(|x| *x <<= k_rem);
            })
        }
    }

    pub fn normalize(&mut self, carry: &mut [i64]) {
        assert!(
            carry.len() >= self.n,
            "invalid carry: carry.len()={} < self.n()={}",
            carry.len(),
            self.n()
        );
        unsafe {
            znx_zero_i64_ref(self.n() as u64, carry.as_mut_ptr());
            (0..self.limbs()).rev().for_each(|i| {
                znx_normalize(
                    self.n as u64,
                    self.k as u64,
                    self.at_mut_ptr(i),
                    carry.as_mut_ptr(),
                    self.at_mut_ptr(i),
                    carry.as_mut_ptr(),
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
        let rem: usize = self.k - (self.prec % self.k);
        (1..self.limbs()).for_each(|i| {
            if i == self.limbs() - 1 && rem != self.k {
                let k_rem: usize = self.k - rem;
                izip!(self.at(i).iter(), data.iter_mut()).for_each(|(x, y)| {
                    *y = (*y << k_rem) + (x >> rem);
                });
            } else {
                izip!(self.at(i).iter(), data.iter_mut()).for_each(|(x, y)| {
                    *y = (*y << self.k) + x;
                });
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::poly::Poly;
    use itertools::izip;
    use sampling::source::Source;

    #[test]
    fn test_set_get_i64_lo_norm() {
        let n: usize = 32;
        let k: usize = 19;
        let prec: usize = 128;
        let mut a: Poly = Poly::new(n, k, prec);
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
        let n: usize = 1;
        let k: usize = 19;
        let prec: usize = 128;
        let mut a: Poly = Poly::new(n, k, prec);
        let mut have: Vec<i64> = vec![i64::default(); n];
        let mut source = Source::new([1; 32]);
        have.iter_mut().for_each(|x| {
            *x = source
                .next_u64n(u64::MAX, u64::MAX)
                .wrapping_sub(u64::MAX / 2 + 1) as i64;
        });
        a.set_i64(&have, 63);
        let mut want = vec![i64::default(); n];
        a.get_i64(&mut want);
        izip!(want, have).for_each(|(a, b)| assert_eq!(a, b));
    }
}
