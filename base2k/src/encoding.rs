use crate::ffi::znx::znx_zero_i64_ref;
use crate::{Infos, VecZnx};
use itertools::izip;
use std::cmp::min;

pub trait Encoding {
    /// encode a vector of i64 on the receiver.
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: base two logarithm decomposition of the receiver.
    /// * `log_k`: base two logarithm of the scaling of the data.
    /// * `data`: data to encode on the receiver.
    /// * `log_max`: base two logarithm of the infinity norm of the input data.
    fn encode_vec_i64(&mut self, log_base2k: usize, log_k: usize, data: &[i64], log_max: usize);

    /// decode a vector of i64 from the receiver.
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: base two logarithm decomposition of the receiver.
    /// * `log_k`: base two logarithm of the scaling of the data.
    /// * `data`: data to decode from the receiver.
    fn decode_vec_i64(&self, log_base2k: usize, log_k: usize, data: &mut [i64]);

    /// encodes a single i64 on the receiver at the given index.
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: base two logarithm decomposition of the receiver.
    /// * `log_k`: base two logarithm of the scaling of the data.
    /// * `i`: index of the coefficient on which to encode the data.
    /// * `data`: data to encode on the receiver.
    /// * `log_max`: base two logarithm of the infinity norm of the input data.
    fn encode_coeff_i64(
        &mut self,
        log_base2k: usize,
        log_k: usize,
        i: usize,
        data: i64,
        log_max: usize,
    );

    /// decode a single of i64 from the receiver at the given index.
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: base two logarithm decomposition of the receiver.
    /// * `log_k`: base two logarithm of the scaling of the data.
    /// * `i`: index of the coefficient to decode.
    /// * `data`: data to decode from the receiver.
    fn decode_coeff_i64(&self, log_base2k: usize, log_k: usize, i: usize) -> i64;
}

impl Encoding for VecZnx {
    fn encode_vec_i64(&mut self, log_base2k: usize, log_k: usize, data: &[i64], log_max: usize) {
        let limbs: usize = (log_k + log_base2k - 1) / log_base2k;

        println!("limbs: {}", limbs);

        assert!(limbs <= self.limbs(), "invalid argument log_k: (log_k + self.log_base2k - 1)/self.log_base2k={} > self.limbs()={}", limbs, self.limbs());

        let size: usize = min(data.len(), self.n());
        let log_k_rem: usize = log_base2k - (log_k % log_base2k);

        // If 2^{log_base2k} * 2^{k_rem} < 2^{63}-1, then we can simply copy
        // values on the last limb.
        // Else we decompose values base2k.
        if log_max + log_k_rem < 63 || log_k_rem == log_base2k {
            (0..self.limbs()).for_each(|i| unsafe {
                znx_zero_i64_ref(size as u64, self.at_mut(i).as_mut_ptr());
            });
            self.at_mut(limbs - 1)[..size].copy_from_slice(&data[..size]);
        } else {
            let mask: i64 = (1 << log_base2k) - 1;
            let steps: usize = min(limbs, (log_max + log_base2k - 1) / log_base2k);

            (0..steps).for_each(|i| unsafe {
                znx_zero_i64_ref(size as u64, self.at_mut(i).as_mut_ptr());
            });

            (limbs - steps..limbs)
                .rev()
                .enumerate()
                .for_each(|(i, i_rev)| {
                    let shift: usize = i * log_base2k;
                    izip!(self.at_mut(i_rev)[..size].iter_mut(), data[..size].iter())
                        .for_each(|(y, x)| *y = (x >> shift) & mask);
                })
        }

        // Case where self.prec % self.k != 0.
        if log_k_rem != log_base2k {
            let limbs = self.limbs();
            let steps: usize = min(limbs, (log_max + log_base2k - 1) / log_base2k);
            (limbs - steps..limbs).rev().for_each(|i| {
                self.at_mut(i)[..size]
                    .iter_mut()
                    .for_each(|x| *x <<= log_k_rem);
            })
        }
    }

    fn decode_vec_i64(&self, log_base2k: usize, log_k: usize, data: &mut [i64]) {
        let limbs: usize = (log_k + log_base2k - 1) / log_base2k;
        assert!(
            data.len() >= self.n,
            "invalid data: data.len()={} < self.n()={}",
            data.len(),
            self.n
        );
        data.copy_from_slice(self.at(0));
        let rem: usize = log_base2k - (log_k % log_base2k);
        (1..limbs).for_each(|i| {
            if i == limbs - 1 && rem != log_base2k {
                let k_rem: usize = log_base2k - rem;
                izip!(self.at(i).iter(), data.iter_mut()).for_each(|(x, y)| {
                    *y = (*y << k_rem) + (x >> rem);
                });
            } else {
                izip!(self.at(i).iter(), data.iter_mut()).for_each(|(x, y)| {
                    *y = (*y << log_base2k) + x;
                });
            }
        })
    }

    fn encode_coeff_i64(
        &mut self,
        log_base2k: usize,
        log_k: usize,
        i: usize,
        value: i64,
        log_max: usize,
    ) {
        assert!(i < self.n());
        let limbs: usize = (log_k + log_base2k - 1) / log_base2k;
        assert!(limbs <= self.limbs(), "invalid argument log_k: (log_k + self.log_base2k - 1)/self.log_base2k={} > self.limbs()={}", limbs, self.limbs());
        let log_k_rem: usize = log_base2k - (log_k % log_base2k);
        let limbs = self.limbs();

        // If 2^{log_base2k} * 2^{log_k_rem} < 2^{63}-1, then we can simply copy
        // values on the last limb.
        // Else we decompose values base2k.
        if log_max + log_k_rem < 63 || log_k_rem == log_base2k {
            (0..limbs - 1).for_each(|j| self.at_mut(j)[i] = 0);

            self.at_mut(self.limbs() - 1)[i] = value;
        } else {
            let mask: i64 = (1 << log_base2k) - 1;
            let steps: usize = min(limbs, (log_max + log_base2k - 1) / log_base2k);

            (0..limbs - steps).for_each(|j| self.at_mut(j)[i] = 0);

            (limbs - steps..limbs)
                .rev()
                .enumerate()
                .for_each(|(j, j_rev)| {
                    self.at_mut(j_rev)[i] = (value >> (j * log_base2k)) & mask;
                })
        }

        // Case where self.prec % self.k != 0.
        if log_k_rem != log_base2k {
            let limbs = self.limbs();
            let steps: usize = min(limbs, (log_max + log_base2k - 1) / log_base2k);
            (limbs - steps..limbs).rev().for_each(|j| {
                self.at_mut(j)[i] <<= log_k_rem;
            })
        }
    }

    fn decode_coeff_i64(&self, log_base2k: usize, log_k: usize, i: usize) -> i64 {
        let limbs: usize = (log_k + log_base2k - 1) / log_base2k;
        assert!(i < self.n());
        let mut res: i64 = self.data[i];
        let rem: usize = log_base2k - (log_k % log_base2k);
        (1..limbs).for_each(|i| {
            let x = self.data[i * self.n];
            if i == limbs - 1 && rem != log_base2k {
                let k_rem: usize = log_base2k - rem;
                res = (res << k_rem) + (x >> rem);
            } else {
                res = (res << log_base2k) + x;
            }
        });
        res
    }
}

#[cfg(test)]
mod tests {
    use crate::{Encoding, VecZnx};
    use itertools::izip;
    use sampling::source::Source;

    #[test]
    fn test_set_get_i64_lo_norm() {
        let n: usize = 8;
        let log_base2k: usize = 17;
        let limbs: usize = 5;
        let log_k: usize = limbs * log_base2k - 5;
        let mut a: VecZnx = VecZnx::new(n, limbs);
        let mut have: Vec<i64> = vec![i64::default(); n];
        have.iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = (i as i64) - (n as i64) / 2);
        a.encode_vec_i64(log_base2k, log_k, &have, 10);
        let mut want = vec![i64::default(); n];
        a.decode_vec_i64(log_base2k, log_k, &mut want);
        izip!(want, have).for_each(|(a, b)| assert_eq!(a, b));
    }

    #[test]
    fn test_set_get_i64_hi_norm() {
        let n: usize = 8;
        let log_base2k: usize = 17;
        let limbs: usize = 5;
        let log_k: usize = limbs * log_base2k - 5;
        let mut a: VecZnx = VecZnx::new(n, limbs);
        let mut have: Vec<i64> = vec![i64::default(); n];
        let mut source = Source::new([1; 32]);
        have.iter_mut().for_each(|x| {
            *x = source
                .next_u64n(u64::MAX, u64::MAX)
                .wrapping_sub(u64::MAX / 2 + 1) as i64;
        });
        a.encode_vec_i64(log_base2k, log_k, &have, 63);
        //(0..a.limbs()).for_each(|i| println!("i:{} -> {:?}", i, a.at(i)));
        let mut want = vec![i64::default(); n];
        //(0..a.limbs()).for_each(|i| println!("i:{} -> {:?}", i, a.at(i)));
        a.decode_vec_i64(log_base2k, log_k, &mut want);
        izip!(want, have).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
    }
}
