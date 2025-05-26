use crate::ffi::znx::znx_zero_i64_ref;
use crate::znx_base::{ZnxView, ZnxViewMut};
use crate::{VecZnx, znx_base::ZnxInfos};
use itertools::izip;
use rug::{Assign, Float};
use std::cmp::min;

pub trait Encoding {
    /// encode a vector of i64 on the receiver.
    ///
    /// # Arguments
    ///
    /// * `col_i`: the index of the poly where to encode the data.
    /// * `basek`: base two negative logarithm decomposition of the receiver.
    /// * `k`: base two negative logarithm of the scaling of the data.
    /// * `data`: data to encode on the receiver.
    /// * `log_max`: base two logarithm of the infinity norm of the input data.
    fn encode_vec_i64(&mut self, col_i: usize, basek: usize, k: usize, data: &[i64], log_max: usize);

    /// encodes a single i64 on the receiver at the given index.
    ///
    /// # Arguments
    ///
    /// * `col_i`: the index of the poly where to encode the data.
    /// * `basek`: base two negative logarithm decomposition of the receiver.
    /// * `k`: base two negative logarithm of the scaling of the data.
    /// * `i`: index of the coefficient on which to encode the data.
    /// * `data`: data to encode on the receiver.
    /// * `log_max`: base two logarithm of the infinity norm of the input data.
    fn encode_coeff_i64(&mut self, col_i: usize, basek: usize, k: usize, i: usize, data: i64, log_max: usize);
}

pub trait Decoding {
    /// decode a vector of i64 from the receiver.
    ///
    /// # Arguments
    ///
    /// * `col_i`: the index of the poly where to encode the data.
    /// * `basek`: base two negative logarithm decomposition of the receiver.
    /// * `k`: base two logarithm of the scaling of the data.
    /// * `data`: data to decode from the receiver.
    fn decode_vec_i64(&self, col_i: usize, basek: usize, k: usize, data: &mut [i64]);

    /// decode a vector of Float from the receiver.
    ///
    /// # Arguments
    /// * `col_i`: the index of the poly where to encode the data.
    /// * `basek`: base two negative logarithm decomposition of the receiver.
    /// * `data`: data to decode from the receiver.
    fn decode_vec_float(&self, col_i: usize, basek: usize, data: &mut [Float]);

    /// decode a single of i64 from the receiver at the given index.
    ///
    /// # Arguments
    ///
    /// * `col_i`: the index of the poly where to encode the data.
    /// * `basek`: base two negative logarithm decomposition of the receiver.
    /// * `k`: base two negative logarithm of the scaling of the data.
    /// * `i`: index of the coefficient to decode.
    /// * `data`: data to decode from the receiver.
    fn decode_coeff_i64(&self, col_i: usize, basek: usize, k: usize, i: usize) -> i64;
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> Encoding for VecZnx<D> {
    fn encode_vec_i64(&mut self, col_i: usize, basek: usize, k: usize, data: &[i64], log_max: usize) {
        encode_vec_i64(self, col_i, basek, k, data, log_max)
    }

    fn encode_coeff_i64(&mut self, col_i: usize, basek: usize, k: usize, i: usize, value: i64, log_max: usize) {
        encode_coeff_i64(self, col_i, basek, k, i, value, log_max)
    }
}

impl<D: AsRef<[u8]>> Decoding for VecZnx<D> {
    fn decode_vec_i64(&self, col_i: usize, basek: usize, k: usize, data: &mut [i64]) {
        decode_vec_i64(self, col_i, basek, k, data)
    }

    fn decode_vec_float(&self, col_i: usize, basek: usize, data: &mut [Float]) {
        decode_vec_float(self, col_i, basek, data)
    }

    fn decode_coeff_i64(&self, col_i: usize, basek: usize, k: usize, i: usize) -> i64 {
        decode_coeff_i64(self, col_i, basek, k, i)
    }
}

fn encode_vec_i64<D: AsMut<[u8]> + AsRef<[u8]>>(
    a: &mut VecZnx<D>,
    col_i: usize,
    basek: usize,
    k: usize,
    data: &[i64],
    log_max: usize,
) {
    let size: usize = (k + basek - 1) / basek;

    #[cfg(debug_assertions)]
    {
        assert!(
            size <= a.size(),
            "invalid argument k: (k + a.basek - 1)/a.basek={} > a.size()={}",
            size,
            a.size()
        );
        assert!(col_i < a.cols());
        assert!(data.len() <= a.n())
    }

    let data_len: usize = data.len();
    let k_rem: usize = basek - (k % basek);

    // Zeroes coefficients of the i-th column
    (0..a.size()).for_each(|i| unsafe {
        znx_zero_i64_ref(a.n() as u64, a.at_mut_ptr(col_i, i));
    });

    // If 2^{basek} * 2^{k_rem} < 2^{63}-1, then we can simply copy
    // values on the last limb.
    // Else we decompose values base2k.
    if log_max + k_rem < 63 || k_rem == basek {
        a.at_mut(col_i, size - 1)[..data_len].copy_from_slice(&data[..data_len]);
    } else {
        let mask: i64 = (1 << basek) - 1;
        let steps: usize = min(size, (log_max + basek - 1) / basek);
        (size - steps..size)
            .rev()
            .enumerate()
            .for_each(|(i, i_rev)| {
                let shift: usize = i * basek;
                izip!(a.at_mut(col_i, i_rev).iter_mut(), data.iter()).for_each(|(y, x)| *y = (x >> shift) & mask);
            })
    }

    // Case where self.prec % self.k != 0.
    if k_rem != basek {
        let steps: usize = min(size, (log_max + basek - 1) / basek);
        (size - steps..size).rev().for_each(|i| {
            a.at_mut(col_i, i)[..data_len]
                .iter_mut()
                .for_each(|x| *x <<= k_rem);
        })
    }
}

fn decode_vec_i64<D: AsRef<[u8]>>(a: &VecZnx<D>, col_i: usize, basek: usize, k: usize, data: &mut [i64]) {
    let size: usize = (k + basek - 1) / basek;
    #[cfg(debug_assertions)]
    {
        assert!(
            data.len() >= a.n(),
            "invalid data: data.len()={} < a.n()={}",
            data.len(),
            a.n()
        );
        assert!(col_i < a.cols());
    }
    data.copy_from_slice(a.at(col_i, 0));
    let rem: usize = basek - (k % basek);
    (1..size).for_each(|i| {
        if i == size - 1 && rem != basek {
            let k_rem: usize = basek - rem;
            izip!(a.at(col_i, i).iter(), data.iter_mut()).for_each(|(x, y)| {
                *y = (*y << k_rem) + (x >> rem);
            });
        } else {
            izip!(a.at(col_i, i).iter(), data.iter_mut()).for_each(|(x, y)| {
                *y = (*y << basek) + x;
            });
        }
    })
}

fn decode_vec_float<D: AsRef<[u8]>>(a: &VecZnx<D>, col_i: usize, basek: usize, data: &mut [Float]) {
    let size: usize = a.size();
    #[cfg(debug_assertions)]
    {
        assert!(
            data.len() >= a.n(),
            "invalid data: data.len()={} < a.n()={}",
            data.len(),
            a.n()
        );
        assert!(col_i < a.cols());
    }

    let prec: u32 = (basek * size) as u32;

    // 2^{basek}
    let base = Float::with_val(prec, (1 << basek) as f64);

    // y[i] = sum x[j][i] * 2^{-basek*j}
    (0..size).for_each(|i| {
        if i == 0 {
            izip!(a.at(col_i, size - i - 1).iter(), data.iter_mut()).for_each(|(x, y)| {
                y.assign(*x);
                *y /= &base;
            });
        } else {
            izip!(a.at(col_i, size - i - 1).iter(), data.iter_mut()).for_each(|(x, y)| {
                *y += Float::with_val(prec, *x);
                *y /= &base;
            });
        }
    });
}

fn encode_coeff_i64<D: AsMut<[u8]> + AsRef<[u8]>>(
    a: &mut VecZnx<D>,
    col_i: usize,
    basek: usize,
    k: usize,
    i: usize,
    value: i64,
    log_max: usize,
) {
    let size: usize = (k + basek - 1) / basek;

    #[cfg(debug_assertions)]
    {
        assert!(i < a.n());
        assert!(
            size <= a.size(),
            "invalid argument k: (k + a.basek - 1)/a.basek={} > a.size()={}",
            size,
            a.size()
        );
        assert!(col_i < a.cols());
    }

    let k_rem: usize = basek - (k % basek);
    (0..a.size()).for_each(|j| a.at_mut(col_i, j)[i] = 0);

    // If 2^{basek} * 2^{k_rem} < 2^{63}-1, then we can simply copy
    // values on the last limb.
    // Else we decompose values base2k.
    if log_max + k_rem < 63 || k_rem == basek {
        a.at_mut(col_i, size - 1)[i] = value;
    } else {
        let mask: i64 = (1 << basek) - 1;
        let steps: usize = min(size, (log_max + basek - 1) / basek);
        (size - steps..size)
            .rev()
            .enumerate()
            .for_each(|(j, j_rev)| {
                a.at_mut(col_i, j_rev)[i] = (value >> (j * basek)) & mask;
            })
    }

    // Case where prec % k != 0.
    if k_rem != basek {
        let steps: usize = min(size, (log_max + basek - 1) / basek);
        (size - steps..size).rev().for_each(|j| {
            a.at_mut(col_i, j)[i] <<= k_rem;
        })
    }
}

fn decode_coeff_i64<D: AsRef<[u8]>>(a: &VecZnx<D>, col_i: usize, basek: usize, k: usize, i: usize) -> i64 {
    #[cfg(debug_assertions)]
    {
        assert!(i < a.n());
        assert!(col_i < a.cols())
    }

    let size: usize = (k + basek - 1) / basek;
    let data: &[i64] = a.raw();
    let mut res: i64 = data[i];
    let rem: usize = basek - (k % basek);
    let slice_size: usize = a.n() * a.size();
    (1..size).for_each(|i| {
        let x: i64 = data[i * slice_size];
        if i == size - 1 && rem != basek {
            let k_rem: usize = basek - rem;
            res = (res << k_rem) + (x >> rem);
        } else {
            res = (res << basek) + x;
        }
    });
    res
}

#[cfg(test)]
mod tests {
    use crate::vec_znx_ops::*;
    use crate::znx_base::*;
    use crate::{Decoding, Encoding, FFT64, Module, VecZnx, znx_base::ZnxInfos};
    use itertools::izip;
    use sampling::source::Source;

    #[test]
    fn test_set_get_i64_lo_norm() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let basek: usize = 17;
        let size: usize = 5;
        let k: usize = size * basek - 5;
        let mut a: VecZnx<_> = module.new_vec_znx(2, size);
        let mut source: Source = Source::new([0u8; 32]);
        let raw: &mut [i64] = a.raw_mut();
        raw.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);
        (0..a.cols()).for_each(|col_i| {
            let mut have: Vec<i64> = vec![i64::default(); n];
            have.iter_mut()
                .for_each(|x| *x = (source.next_i64() << 56) >> 56);
            a.encode_vec_i64(col_i, basek, k, &have, 10);
            let mut want: Vec<i64> = vec![i64::default(); n];
            a.decode_vec_i64(col_i, basek, k, &mut want);
            izip!(want, have).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
        });
    }

    #[test]
    fn test_set_get_i64_hi_norm() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let basek: usize = 17;
        let size: usize = 5;
        let k: usize = size * basek - 5;
        let mut a: VecZnx<_> = module.new_vec_znx(2, size);
        let mut source = Source::new([0u8; 32]);
        let raw: &mut [i64] = a.raw_mut();
        raw.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);
        (0..a.cols()).for_each(|col_i| {
            let mut have: Vec<i64> = vec![i64::default(); n];
            have.iter_mut().for_each(|x| *x = source.next_i64());
            a.encode_vec_i64(col_i, basek, k, &have, 64);
            let mut want = vec![i64::default(); n];
            a.decode_vec_i64(col_i, basek, k, &mut want);
            izip!(want, have).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
        })
    }
}
