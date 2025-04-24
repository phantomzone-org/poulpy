use crate::ffi::znx::znx_zero_i64_ref;
use crate::{Infos, VecZnx};
use itertools::izip;
use rug::{Assign, Float};
use std::cmp::min;

pub trait Encoding {
    /// encode a vector of i64 on the receiver.
    ///
    /// # Arguments
    ///
    /// * `poly_idx`: the index of the poly where to encode the data.
    /// * `log_base2k`: base two negative logarithm decomposition of the receiver.
    /// * `log_k`: base two negative logarithm of the scaling of the data.
    /// * `data`: data to encode on the receiver.
    /// * `log_max`: base two logarithm of the infinity norm of the input data.
    fn encode_vec_i64(&mut self, poly_idx: usize, log_base2k: usize, log_k: usize, data: &[i64], log_max: usize);

    /// decode a vector of i64 from the receiver.
    ///
    /// # Arguments
    ///
    /// * `poly_idx`: the index of the poly where to encode the data.
    /// * `log_base2k`: base two negative logarithm decomposition of the receiver.
    /// * `log_k`: base two logarithm of the scaling of the data.
    /// * `data`: data to decode from the receiver.
    fn decode_vec_i64(&self, poly_idx: usize, log_base2k: usize, log_k: usize, data: &mut [i64]);

    /// decode a vector of Float from the receiver.
    ///
    /// # Arguments
    /// * `poly_idx`: the index of the poly where to encode the data.
    /// * `log_base2k`: base two negative logarithm decomposition of the receiver.
    /// * `data`: data to decode from the receiver.
    fn decode_vec_float(&self, poly_idx: usize, log_base2k: usize, data: &mut [Float]);

    /// encodes a single i64 on the receiver at the given index.
    ///
    /// # Arguments
    ///
    /// * `poly_idx`: the index of the poly where to encode the data.
    /// * `log_base2k`: base two negative logarithm decomposition of the receiver.
    /// * `log_k`: base two negative logarithm of the scaling of the data.
    /// * `i`: index of the coefficient on which to encode the data.
    /// * `data`: data to encode on the receiver.
    /// * `log_max`: base two logarithm of the infinity norm of the input data.
    fn encode_coeff_i64(&mut self, poly_idx: usize, log_base2k: usize, log_k: usize, i: usize, data: i64, log_max: usize);

    /// decode a single of i64 from the receiver at the given index.
    ///
    /// # Arguments
    ///
    /// * `poly_idx`: the index of the poly where to encode the data.
    /// * `log_base2k`: base two negative logarithm decomposition of the receiver.
    /// * `log_k`: base two negative logarithm of the scaling of the data.
    /// * `i`: index of the coefficient to decode.
    /// * `data`: data to decode from the receiver.
    fn decode_coeff_i64(&self, poly_idx: usize, log_base2k: usize, log_k: usize, i: usize) -> i64;
}

impl Encoding for VecZnx {
    fn encode_vec_i64(&mut self, poly_idx: usize, log_base2k: usize, log_k: usize, data: &[i64], log_max: usize) {
        encode_vec_i64(self, poly_idx, log_base2k, log_k, data, log_max)
    }

    fn decode_vec_i64(&self, poly_idx: usize, log_base2k: usize, log_k: usize, data: &mut [i64]) {
        decode_vec_i64(self, poly_idx, log_base2k, log_k, data)
    }

    fn decode_vec_float(&self, poly_idx: usize, log_base2k: usize, data: &mut [Float]) {
        decode_vec_float(self, poly_idx, log_base2k, data)
    }

    fn encode_coeff_i64(&mut self, poly_idx: usize, log_base2k: usize, log_k: usize, i: usize, value: i64, log_max: usize) {
        encode_coeff_i64(self, poly_idx, log_base2k, log_k, i, value, log_max)
    }

    fn decode_coeff_i64(&self, poly_idx: usize, log_base2k: usize, log_k: usize, i: usize) -> i64 {
        decode_coeff_i64(self, poly_idx, log_base2k, log_k, i)
    }
}

fn encode_vec_i64(a: &mut VecZnx, poly_idx: usize, log_base2k: usize, log_k: usize, data: &[i64], log_max: usize) {
    let cols: usize = (log_k + log_base2k - 1) / log_base2k;

    #[cfg(debug_assertions)]
    {
        assert!(
            cols <= a.cols(),
            "invalid argument log_k: (log_k + a.log_base2k - 1)/a.log_base2k={} > a.cols()={}",
            cols,
            a.cols()
        );
        assert!(poly_idx < a.size);
        assert!(data.len() <= a.n())
    }

    let data_len: usize = data.len();
    let log_k_rem: usize = log_base2k - (log_k % log_base2k);

    (0..a.cols()).for_each(|i| unsafe {
        znx_zero_i64_ref(a.n() as u64, a.at_poly_mut_ptr(poly_idx, i));
    });

    // If 2^{log_base2k} * 2^{k_rem} < 2^{63}-1, then we can simply copy
    // values on the last limb.
    // Else we decompose values base2k.
    if log_max + log_k_rem < 63 || log_k_rem == log_base2k {
        a.at_poly_mut(poly_idx, cols - 1)[..data_len].copy_from_slice(&data[..data_len]);
    } else {
        let mask: i64 = (1 << log_base2k) - 1;
        let steps: usize = min(cols, (log_max + log_base2k - 1) / log_base2k);
        (cols - steps..cols)
            .rev()
            .enumerate()
            .for_each(|(i, i_rev)| {
                let shift: usize = i * log_base2k;
                izip!(a.at_poly_mut(poly_idx, i_rev).iter_mut(), data.iter()).for_each(|(y, x)| *y = (x >> shift) & mask);
            })
    }

    // Case where self.prec % self.k != 0.
    if log_k_rem != log_base2k {
        let steps: usize = min(cols, (log_max + log_base2k - 1) / log_base2k);
        (cols - steps..cols).rev().for_each(|i| {
            a.at_poly_mut(poly_idx, i)[..data_len]
                .iter_mut()
                .for_each(|x| *x <<= log_k_rem);
        })
    }
}

fn decode_vec_i64(a: &VecZnx, poly_idx: usize, log_base2k: usize, log_k: usize, data: &mut [i64]) {
    let cols: usize = (log_k + log_base2k - 1) / log_base2k;
    #[cfg(debug_assertions)]
    {
        assert!(
            data.len() >= a.n(),
            "invalid data: data.len()={} < a.n()={}",
            data.len(),
            a.n()
        );
        assert!(poly_idx < a.size());
    }
    data.copy_from_slice(a.at_poly(poly_idx, 0));
    let rem: usize = log_base2k - (log_k % log_base2k);
    (1..cols).for_each(|i| {
        if i == cols - 1 && rem != log_base2k {
            let k_rem: usize = log_base2k - rem;
            izip!(a.at_poly(poly_idx, i).iter(), data.iter_mut()).for_each(|(x, y)| {
                *y = (*y << k_rem) + (x >> rem);
            });
        } else {
            izip!(a.at_poly(poly_idx, i).iter(), data.iter_mut()).for_each(|(x, y)| {
                *y = (*y << log_base2k) + x;
            });
        }
    })
}

fn decode_vec_float(a: &VecZnx, poly_idx: usize, log_base2k: usize, data: &mut [Float]) {
    let cols: usize = a.cols();
    #[cfg(debug_assertions)]
    {
        assert!(
            data.len() >= a.n(),
            "invalid data: data.len()={} < a.n()={}",
            data.len(),
            a.n()
        );
        assert!(poly_idx < a.size());
    }

    let prec: u32 = (log_base2k * cols) as u32;

    // 2^{log_base2k}
    let base = Float::with_val(prec, (1 << log_base2k) as f64);

    // y[i] = sum x[j][i] * 2^{-log_base2k*j}
    (0..cols).for_each(|i| {
        if i == 0 {
            izip!(a.at_poly(poly_idx, cols - i - 1).iter(), data.iter_mut()).for_each(|(x, y)| {
                y.assign(*x);
                *y /= &base;
            });
        } else {
            izip!(a.at_poly(poly_idx, cols - i - 1).iter(), data.iter_mut()).for_each(|(x, y)| {
                *y += Float::with_val(prec, *x);
                *y /= &base;
            });
        }
    });
}

fn encode_coeff_i64(a: &mut VecZnx, poly_idx: usize, log_base2k: usize, log_k: usize, i: usize, value: i64, log_max: usize) {
    let cols: usize = (log_k + log_base2k - 1) / log_base2k;

    #[cfg(debug_assertions)]
    {
        assert!(i < a.n());
        assert!(
            cols <= a.cols(),
            "invalid argument log_k: (log_k + a.log_base2k - 1)/a.log_base2k={} > a.cols()={}",
            cols,
            a.cols()
        );
        assert!(poly_idx < a.size());
    }

    let log_k_rem: usize = log_base2k - (log_k % log_base2k);
    (0..a.cols()).for_each(|j| a.at_poly_mut(poly_idx, j)[i] = 0);

    // If 2^{log_base2k} * 2^{log_k_rem} < 2^{63}-1, then we can simply copy
    // values on the last limb.
    // Else we decompose values base2k.
    if log_max + log_k_rem < 63 || log_k_rem == log_base2k {
        a.at_poly_mut(poly_idx, cols - 1)[i] = value;
    } else {
        let mask: i64 = (1 << log_base2k) - 1;
        let steps: usize = min(cols, (log_max + log_base2k - 1) / log_base2k);
        (cols - steps..cols)
            .rev()
            .enumerate()
            .for_each(|(j, j_rev)| {
                a.at_poly_mut(poly_idx, j_rev)[i] = (value >> (j * log_base2k)) & mask;
            })
    }

    // Case where prec % k != 0.
    if log_k_rem != log_base2k {
        let steps: usize = min(cols, (log_max + log_base2k - 1) / log_base2k);
        (cols - steps..cols).rev().for_each(|j| {
            a.at_poly_mut(poly_idx, j)[i] <<= log_k_rem;
        })
    }
}

fn decode_coeff_i64(a: &VecZnx, poly_idx: usize, log_base2k: usize, log_k: usize, i: usize) -> i64 {
    #[cfg(debug_assertions)]
    {
        assert!(i < a.n());
        assert!(poly_idx < a.size())
    }

    let cols: usize = (log_k + log_base2k - 1) / log_base2k;
    let data: &[i64] = a.raw();
    let mut res: i64 = data[i];
    let rem: usize = log_base2k - (log_k % log_base2k);
    let slice_size: usize = a.n() * a.size();
    (1..cols).for_each(|i| {
        let x = data[i * slice_size];
        if i == cols - 1 && rem != log_base2k {
            let k_rem: usize = log_base2k - rem;
            res = (res << k_rem) + (x >> rem);
        } else {
            res = (res << log_base2k) + x;
        }
    });
    res
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
        let cols: usize = 5;
        let log_k: usize = cols * log_base2k - 5;
        let mut a: VecZnx = VecZnx::new(n, 2, cols);
        let mut source: Source = Source::new([0u8; 32]);
        let raw: &mut [i64] = a.raw_mut();
        raw.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);
        (0..a.size()).for_each(|poly_idx| {
            let mut have: Vec<i64> = vec![i64::default(); n];
            have.iter_mut()
                .for_each(|x| *x = (source.next_i64() << 56) >> 56);
            a.encode_vec_i64(poly_idx, log_base2k, log_k, &have, 10);
            let mut want: Vec<i64> = vec![i64::default(); n];
            a.decode_vec_i64(poly_idx, log_base2k, log_k, &mut want);
            izip!(want, have).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
        });
    }

    #[test]
    fn test_set_get_i64_hi_norm() {
        let n: usize = 8;
        let log_base2k: usize = 17;
        let cols: usize = 5;
        let log_k: usize = cols * log_base2k - 5;
        let mut a: VecZnx = VecZnx::new(n, 2, cols);
        let mut source = Source::new([0u8; 32]);
        let raw: &mut [i64] = a.raw_mut();
        raw.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);
        (0..a.size()).for_each(|poly_idx| {
            let mut have: Vec<i64> = vec![i64::default(); n];
            have.iter_mut().for_each(|x| *x = source.next_i64());
            a.encode_vec_i64(poly_idx, log_base2k, log_k, &have, 64);
            let mut want = vec![i64::default(); n];
            a.decode_vec_i64(poly_idx, log_base2k, log_k, &mut want);
            izip!(want, have).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
        })
    }
}
