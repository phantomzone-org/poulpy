use itertools::izip;

use crate::layouts::{VecZnx, VecZnxToMut, ZnxInfos, ZnxViewMut, ZnxZero};

pub fn vec_znx_lsh_inplace_ref<A>(basek: usize, k: usize, a: &mut A, a_col: usize)
where
    A: VecZnxToMut,
{
    let mut a: VecZnx<&mut [u8]> = a.to_mut();

    let n: usize = a.n();
    let cols: usize = a.cols();
    let size: usize = a.size();
    let steps: usize = k / basek;

    if steps > 0 {
        if steps >= size {
            (0..size).for_each(|j| {
                a.zero_at(a_col, j);
            });
            return;
        }

        let start: usize = n * a_col;
        let end: usize = start + n;
        let offset: usize = n * cols * steps;

        let a_raw: &mut [i64] = a.raw_mut();
        (0..size - steps).for_each(|j| {
            let (lhs, rhs) = a_raw.split_at_mut(n * cols * j);
            lhs[start..end].copy_from_slice(&rhs[start + offset..end + offset]);
        });

        (size - steps..size).for_each(|j| {
            a.zero_at(a_col, j);
        });
    }

    let k_rem: usize = k % basek;

    if k_rem != 0 {
        let shift: usize = i64::BITS as usize - k_rem;
        (0..steps).for_each(|j| {
            a.at_mut(a_col, j).iter_mut().for_each(|xi| {
                *xi <<= shift;
            });
        });
    }
}

pub fn vec_znx_rsh_inplace_ref<A>(basek: usize, k: usize, a: &mut A, a_col: usize)
where
    A: VecZnxToMut,
{
    let mut a: VecZnx<&mut [u8]> = a.to_mut();
    let n: usize = a.n();
    let cols: usize = a.cols();
    let size: usize = a.size();
    let steps: usize = k / basek;

    if steps > 0 {
        if steps >= size {
            (0..size).for_each(|j| {
                a.zero_at(a_col, j);
            });
            return;
        }

        let start: usize = n * a_col;
        let end: usize = start + n;
        let offset: usize = n * cols * steps;

        let a_raw: &mut [i64] = a.raw_mut();
        (size - steps..size).rev().for_each(|j| {
            let (lhs, rhs) = a_raw.split_at_mut(n * cols * j);
            rhs[start + offset..end + offset].copy_from_slice(&lhs[start..end]);
        });

        (0..steps).for_each(|j| {
            a.zero_at(a_col, j);
        });
    }

    let k_rem: usize = k % basek;

    if k_rem != 0 {
        let mut carry: Vec<i64> = vec![0i64; n]; // ALLOC (but small so OK)
        let shift: usize = i64::BITS as usize - k_rem;
        carry.fill(0);
        (steps..size).for_each(|j| {
            izip!(carry.iter_mut(), a.at_mut(a_col, j).iter_mut()).for_each(|(ci, xi)| {
                *xi += *ci << basek;
                *ci = (*xi << shift) >> shift;
                *xi = (*xi - *ci) >> k_rem;
            });
        });
    }
}
