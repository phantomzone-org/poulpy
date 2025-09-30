use itertools::izip;
use rug::{Assign, Float};

use crate::{
    layouts::{DataMut, DataRef, VecZnx, VecZnxToMut, VecZnxToRef, Zn, ZnToMut, ZnToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{
        ZnxNormalizeFinalStepInplace, ZnxNormalizeFirstStepInplace, ZnxNormalizeMiddleStepInplace, ZnxRef, ZnxZero,
        get_carry_i128, get_digit_i128, znx_zero_ref,
    },
};

impl<D: DataMut> VecZnx<D> {
    pub fn encode_vec_i64(&mut self, base2k: usize, col: usize, k: usize, data: &[i64]) {
        let size: usize = k.div_ceil(base2k);

        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&mut [u8]> = self.to_mut();
            assert!(
                size <= a.size(),
                "invalid argument k.div_ceil(base2k)={} > a.size()={}",
                size,
                a.size()
            );
            assert!(col < a.cols());
            assert!(data.len() == a.n())
        }

        let mut a: VecZnx<&mut [u8]> = self.to_mut();
        let a_size: usize = a.size();

        // Zeroes coefficients of the i-th column
        for i in 0..a_size {
            znx_zero_ref(a.at_mut(col, i));
        }

        // Copies the data on the correct limb
        a.at_mut(col, size - 1).copy_from_slice(data);

        let mut carry: Vec<i64> = vec![0i64; a.n()];
        let k_rem: usize = (base2k - (k % base2k)) % base2k;

        // Normalizes and shift if necessary.
        for j in (0..size).rev() {
            if j == size - 1 {
                ZnxRef::znx_normalize_first_step_inplace(base2k, k_rem, a.at_mut(col, j), &mut carry);
            } else if j == 0 {
                ZnxRef::znx_normalize_final_step_inplace(base2k, k_rem, a.at_mut(col, j), &mut carry);
            } else {
                ZnxRef::znx_normalize_middle_step_inplace(base2k, k_rem, a.at_mut(col, j), &mut carry);
            }
        }
    }

    pub fn encode_vec_i128(&mut self, base2k: usize, col: usize, k: usize, data: &[i128]) {
        let size: usize = k.div_ceil(base2k);

        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&mut [u8]> = self.to_mut();
            assert!(
                size <= a.size(),
                "invalid argument k.div_ceil(base2k)={} > a.size()={}",
                size,
                a.size()
            );
            assert!(col < a.cols());
            assert!(data.len() == a.n())
        }

        let mut a: VecZnx<&mut [u8]> = self.to_mut();
        let a_size: usize = a.size();

        {
            let mut carry_i128: Vec<i128> = vec![0i128; a.n()];
            carry_i128.copy_from_slice(data);

            for j in (0..size).rev() {
                for (x, a) in izip!(a.at_mut(col, j).iter_mut(), carry_i128.iter_mut()) {
                    let digit: i128 = get_digit_i128(base2k, *a);
                    let carry: i128 = get_carry_i128(base2k, *a, digit);
                    *x = digit as i64;
                    *a = carry;
                }
            }
        }

        for j in size..a_size {
            ZnxRef::znx_zero(a.at_mut(col, j));
        }

        let mut carry: Vec<i64> = vec![0i64; a.n()];
        let k_rem: usize = (base2k - (k % base2k)) % base2k;

        for j in (0..size).rev() {
            if j == a_size - 1 {
                ZnxRef::znx_normalize_first_step_inplace(base2k, k_rem, a.at_mut(col, j), &mut carry);
            } else if j == 0 {
                ZnxRef::znx_normalize_final_step_inplace(base2k, k_rem, a.at_mut(col, j), &mut carry);
            } else {
                ZnxRef::znx_normalize_middle_step_inplace(base2k, k_rem, a.at_mut(col, j), &mut carry);
            }
        }
    }

    pub fn encode_coeff_i64(&mut self, base2k: usize, col: usize, k: usize, idx: usize, data: i64) {
        let size: usize = k.div_ceil(base2k);

        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&mut [u8]> = self.to_mut();
            assert!(idx < a.n());
            assert!(
                size <= a.size(),
                "invalid argument k.div_ceil(base2k)={} > a.size()={}",
                size,
                a.size()
            );
            assert!(col < a.cols());
        }

        let mut a: VecZnx<&mut [u8]> = self.to_mut();
        let a_size = a.size();

        for j in 0..a_size {
            a.at_mut(col, j)[idx] = 0
        }

        a.at_mut(col, size - 1)[idx] = data;

        let mut carry: Vec<i64> = vec![0i64; 1];
        let k_rem: usize = (base2k - (k % base2k)) % base2k;

        for j in (0..size).rev() {
            let slice = &mut a.at_mut(col, j)[idx..idx + 1];

            if j == size - 1 {
                ZnxRef::znx_normalize_first_step_inplace(base2k, k_rem, slice, &mut carry);
            } else if j == 0 {
                ZnxRef::znx_normalize_final_step_inplace(base2k, k_rem, slice, &mut carry);
            } else {
                ZnxRef::znx_normalize_middle_step_inplace(base2k, k_rem, slice, &mut carry);
            }
        }
    }
}

impl<D: DataRef> VecZnx<D> {
    pub fn decode_vec_i64(&self, base2k: usize, col: usize, k: usize, data: &mut [i64]) {
        let size: usize = k.div_ceil(base2k);
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
        let rem: usize = base2k - (k % base2k);
        if k < base2k {
            data.iter_mut().for_each(|x| *x >>= rem);
        } else {
            (1..size).for_each(|i| {
                if i == size - 1 && rem != base2k {
                    let k_rem: usize = (base2k - rem) % base2k;
                    izip!(a.at(col, i).iter(), data.iter_mut()).for_each(|(x, y)| {
                        *y = (*y << k_rem) + (x >> rem);
                    });
                } else {
                    izip!(a.at(col, i).iter(), data.iter_mut()).for_each(|(x, y)| {
                        *y = (*y << base2k) + x;
                    });
                }
            })
        }
    }

    pub fn decode_coeff_i64(&self, base2k: usize, col: usize, k: usize, idx: usize) -> i64 {
        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&[u8]> = self.to_ref();
            assert!(idx < a.n());
            assert!(col < a.cols())
        }

        let a: VecZnx<&[u8]> = self.to_ref();
        let size: usize = k.div_ceil(base2k);
        let mut res: i64 = 0;
        let rem: usize = base2k - (k % base2k);
        (0..size).for_each(|j| {
            let x: i64 = a.at(col, j)[idx];
            if j == size - 1 && rem != base2k {
                let k_rem: usize = (base2k - rem) % base2k;
                res = (res << k_rem) + (x >> rem);
            } else {
                res = (res << base2k) + x;
            }
        });
        res
    }

    pub fn decode_vec_float(&self, base2k: usize, col: usize, data: &mut [Float]) {
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
        let prec: u32 = (base2k * size) as u32;

        // 2^{base2k}
        let base: Float = Float::with_val(prec, (1u64 << base2k) as f64);

        // y[i] = sum x[j][i] * 2^{-base2k*j}
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
    pub fn encode_i64(&mut self, base2k: usize, k: usize, data: i64) {
        let size: usize = k.div_ceil(base2k);

        #[cfg(debug_assertions)]
        {
            let a: Zn<&mut [u8]> = self.to_mut();
            assert!(
                size <= a.size(),
                "invalid argument k.div_ceil(base2k)={} > a.size()={}",
                size,
                a.size()
            );
        }

        let mut a: Zn<&mut [u8]> = self.to_mut();
        let a_size = a.size();

        for j in 0..a_size {
            a.at_mut(0, j)[0] = 0
        }

        a.at_mut(0, size - 1)[0] = data;

        let mut carry: Vec<i64> = vec![0i64; 1];
        let k_rem: usize = (base2k - (k % base2k)) % base2k;

        for j in (0..size).rev() {
            let slice = &mut a.at_mut(0, j)[..1];

            if j == size - 1 {
                ZnxRef::znx_normalize_first_step_inplace(base2k, k_rem, slice, &mut carry);
            } else if j == 0 {
                ZnxRef::znx_normalize_final_step_inplace(base2k, k_rem, slice, &mut carry);
            } else {
                ZnxRef::znx_normalize_middle_step_inplace(base2k, k_rem, slice, &mut carry);
            }
        }
    }
}

impl<D: DataRef> Zn<D> {
    pub fn decode_i64(&self, base2k: usize, k: usize) -> i64 {
        let a: Zn<&[u8]> = self.to_ref();
        let size: usize = k.div_ceil(base2k);
        let mut res: i64 = 0;
        let rem: usize = base2k - (k % base2k);
        (0..size).for_each(|j| {
            let x: i64 = a.at(0, j)[0];
            if j == size - 1 && rem != base2k {
                let k_rem: usize = (base2k - rem) % base2k;
                res = (res << k_rem) + (x >> rem);
            } else {
                res = (res << base2k) + x;
            }
        });
        res
    }

    pub fn decode_float(&self, base2k: usize) -> Float {
        let a: Zn<&[u8]> = self.to_ref();
        let size: usize = a.size();
        let prec: u32 = (base2k * size) as u32;

        // 2^{base2k}
        let base: Float = Float::with_val(prec, (1 << base2k) as f64);
        let mut res: Float = Float::with_val(prec, (1 << base2k) as f64);

        // y[i] = sum x[j][i] * 2^{-base2k*j}
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
