use itertools::izip;
use rug::{Assign, Float};

use crate::{
    layouts::{DataMut, DataRef, VecZnx, VecZnxToMut, VecZnxToRef, Zn, ZnToMut, ZnToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::zero::znx_zero_ref,
};

impl<D: DataMut> VecZnx<D> {
    pub fn encode_vec_i64(&mut self, basek: usize, col: usize, k: usize, data: &[i64], log_max: usize) {
        let size: usize = k.div_ceil(basek);

        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&mut [u8]> = self.to_mut();
            assert!(
                size <= a.size(),
                "invalid argument k.div_ceil(basek)={} > a.size()={}",
                size,
                a.size()
            );
            assert!(col < a.cols());
            assert!(data.len() <= a.n())
        }

        let data_len: usize = data.len();
        let mut a: VecZnx<&mut [u8]> = self.to_mut();
        let k_rem: usize = basek - (k % basek);

        // Zeroes coefficients of the i-th column
        (0..a.size()).for_each(|i| {
            znx_zero_ref(a.at_mut(col, i));
        });

        // If 2^{basek} * 2^{k_rem} < 2^{63}-1, then we can simply copy
        // values on the last limb.
        // Else we decompose values base2k.
        if log_max + k_rem < 63 || k_rem == basek {
            a.at_mut(col, size - 1)[..data_len].copy_from_slice(&data[..data_len]);
        } else {
            let mask: i64 = (1 << basek) - 1;
            let steps: usize = size.min(log_max.div_ceil(basek));
            (size - steps..size)
                .rev()
                .enumerate()
                .for_each(|(i, i_rev)| {
                    let shift: usize = i * basek;
                    izip!(a.at_mut(col, i_rev).iter_mut(), data.iter()).for_each(|(y, x)| *y = (x >> shift) & mask);
                })
        }

        // Case where self.prec % self.k != 0.
        if k_rem != basek {
            let steps: usize = size.min(log_max.div_ceil(basek));
            (size - steps..size).rev().for_each(|i| {
                a.at_mut(col, i)[..data_len]
                    .iter_mut()
                    .for_each(|x| *x <<= k_rem);
            })
        }
    }

    pub fn encode_coeff_i64(&mut self, basek: usize, col: usize, k: usize, idx: usize, data: i64, log_max: usize) {
        let size: usize = k.div_ceil(basek);

        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&mut [u8]> = self.to_mut();
            assert!(idx < a.n());
            assert!(
                size <= a.size(),
                "invalid argument k.div_ceil(basek)={} > a.size()={}",
                size,
                a.size()
            );
            assert!(col < a.cols());
        }

        let k_rem: usize = basek - (k % basek);
        let mut a: VecZnx<&mut [u8]> = self.to_mut();
        (0..a.size()).for_each(|j| a.at_mut(col, j)[idx] = 0);

        // If 2^{basek} * 2^{k_rem} < 2^{63}-1, then we can simply copy
        // values on the last limb.
        // Else we decompose values base2k.
        if log_max + k_rem < 63 || k_rem == basek {
            a.at_mut(col, size - 1)[idx] = data;
        } else {
            let mask: i64 = (1 << basek) - 1;
            let steps: usize = size.min(log_max.div_ceil(basek));
            (size - steps..size)
                .rev()
                .enumerate()
                .for_each(|(j, j_rev)| {
                    a.at_mut(col, j_rev)[idx] = (data >> (j * basek)) & mask;
                })
        }

        // Case where prec % k != 0.
        if k_rem != basek {
            let steps: usize = size.min(log_max.div_ceil(basek));
            (size - steps..size).rev().for_each(|j| {
                a.at_mut(col, j)[idx] <<= k_rem;
            })
        }
    }
}

impl<D: DataRef> VecZnx<D> {
    pub fn decode_vec_i64(&self, basek: usize, col: usize, k: usize, data: &mut [i64]) {
        let size: usize = k.div_ceil(basek);
        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&[u8]> = self.to_ref();
            assert!(
                data.len() >= a.n(),
                "invalid data: data.len()={} < a.n()={}",
                data.len(),
                a.n()
            );
            assert!(col < a.cols());
        }

        let a: VecZnx<&[u8]> = self.to_ref();
        data.copy_from_slice(a.at(col, 0));
        let rem: usize = basek - (k % basek);
        if k < basek {
            data.iter_mut().for_each(|x| *x >>= rem);
        } else {
            (1..size).for_each(|i| {
                if i == size - 1 && rem != basek {
                    let k_rem: usize = basek - rem;
                    izip!(a.at(col, i).iter(), data.iter_mut()).for_each(|(x, y)| {
                        *y = (*y << k_rem) + (x >> rem);
                    });
                } else {
                    izip!(a.at(col, i).iter(), data.iter_mut()).for_each(|(x, y)| {
                        *y = (*y << basek) + x;
                    });
                }
            })
        }
    }

    pub fn decode_coeff_i64(&self, basek: usize, col: usize, k: usize, idx: usize) -> i64 {
        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&[u8]> = self.to_ref();
            assert!(idx < a.n());
            assert!(col < a.cols())
        }

        let a: VecZnx<&[u8]> = self.to_ref();
        let size: usize = k.div_ceil(basek);
        let mut res: i64 = 0;
        let rem: usize = basek - (k % basek);
        (0..size).for_each(|j| {
            let x: i64 = a.at(col, j)[idx];
            if j == size - 1 && rem != basek {
                let k_rem: usize = basek - rem;
                res = (res << k_rem) + (x >> rem);
            } else {
                res = (res << basek) + x;
            }
        });
        res
    }

    pub fn decode_vec_float(&self, basek: usize, col: usize, data: &mut [Float]) {
        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&[u8]> = self.to_ref();
            assert!(
                data.len() >= a.n(),
                "invalid data: data.len()={} < a.n()={}",
                data.len(),
                a.n()
            );
            assert!(col < a.cols());
        }

        let a: VecZnx<&[u8]> = self.to_ref();
        let size: usize = a.size();
        let prec: u32 = (basek * size) as u32;

        // 2^{basek}
        let base = Float::with_val(prec, (1u64 << basek) as f64);

        // y[i] = sum x[j][i] * 2^{-basek*j}
        (0..size).for_each(|i| {
            if i == 0 {
                izip!(a.at(col, size - i - 1).iter(), data.iter_mut()).for_each(|(x, y)| {
                    y.assign(*x);
                    *y /= &base;
                });
            } else {
                izip!(a.at(col, size - i - 1).iter(), data.iter_mut()).for_each(|(x, y)| {
                    *y += Float::with_val(prec, *x);
                    *y /= &base;
                });
            }
        });
    }
}

impl<D: DataMut> Zn<D> {
    pub fn encode_i64(&mut self, basek: usize, k: usize, data: i64, log_max: usize) {
        let size: usize = k.div_ceil(basek);

        #[cfg(debug_assertions)]
        {
            let a: Zn<&mut [u8]> = self.to_mut();
            assert!(
                size <= a.size(),
                "invalid argument k.div_ceil(basek)={} > a.size()={}",
                size,
                a.size()
            );
        }

        let k_rem: usize = basek - (k % basek);
        let mut a: Zn<&mut [u8]> = self.to_mut();
        (0..a.size()).for_each(|j| a.at_mut(0, j)[0] = 0);

        // If 2^{basek} * 2^{k_rem} < 2^{63}-1, then we can simply copy
        // values on the last limb.
        // Else we decompose values base2k.
        if log_max + k_rem < 63 || k_rem == basek {
            a.at_mut(0, size - 1)[0] = data;
        } else {
            let mask: i64 = (1 << basek) - 1;
            let steps: usize = size.min(log_max.div_ceil(basek));
            (size - steps..size)
                .rev()
                .enumerate()
                .for_each(|(j, j_rev)| {
                    a.at_mut(0, j_rev)[0] = (data >> (j * basek)) & mask;
                })
        }

        // Case where prec % k != 0.
        if k_rem != basek {
            let steps: usize = size.min(log_max.div_ceil(basek));
            (size - steps..size).rev().for_each(|j| {
                a.at_mut(0, j)[0] <<= k_rem;
            })
        }
    }
}

impl<D: DataRef> Zn<D> {
    pub fn decode_i64(&self, basek: usize, k: usize) -> i64 {
        let a: Zn<&[u8]> = self.to_ref();
        let size: usize = k.div_ceil(basek);
        let mut res: i64 = 0;
        let rem: usize = basek - (k % basek);
        (0..size).for_each(|j| {
            let x: i64 = a.at(0, j)[0];
            if j == size - 1 && rem != basek {
                let k_rem: usize = basek - rem;
                res = (res << k_rem) + (x >> rem);
            } else {
                res = (res << basek) + x;
            }
        });
        res
    }

    pub fn decode_float(&self, basek: usize) -> Float {
        let a: Zn<&[u8]> = self.to_ref();
        let size: usize = a.size();
        let prec: u32 = (basek * size) as u32;

        // 2^{basek}
        let base: Float = Float::with_val(prec, (1 << basek) as f64);
        let mut res: Float = Float::with_val(prec, (1 << basek) as f64);

        // y[i] = sum x[j][i] * 2^{-basek*j}
        (0..size).for_each(|i| {
            if i == 0 {
                res.assign(a.at(0, size - i - 1)[0]);
                res /= &base;
            } else {
                res += Float::with_val(prec, a.at(0, size - i - 1)[0]);
                res /= &base;
            }
        });

        res
    }
}
