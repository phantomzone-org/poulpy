use poulpy_hal::{
    api::TakeSlice,
    layouts::{Scratch, Zn, ZnToMut, ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut},
    oep::{
        TakeSliceImpl, ZnAddDistF64Impl, ZnAddNormalImpl, ZnFillDistF64Impl, ZnFillNormalImpl, ZnFillUniformImpl,
        ZnNormalizeInplaceImpl,
    },
    source::Source,
};
use rand_distr::Normal;

use crate::cpu_spqlios::{FFT64, ffi::zn64};

unsafe impl ZnNormalizeInplaceImpl<Self> for FFT64
where
    Self: TakeSliceImpl<Self>,
{
    fn zn_normalize_inplace_impl<A>(n: usize, basek: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<Self>)
    where
        A: ZnToMut,
    {
        let mut a: Zn<&mut [u8]> = a.to_mut();

        let (tmp_bytes, _) = scratch.take_slice(n * size_of::<i64>());

        unsafe {
            zn64::zn64_normalize_base2k_ref(
                n as u64,
                basek as u64,
                a.at_mut_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }
}

unsafe impl ZnFillUniformImpl<Self> for FFT64 {
    fn zn_fill_uniform_impl<R>(n: usize, basek: usize, res: &mut R, res_col: usize, k: usize, source: &mut Source)
    where
        R: ZnToMut,
    {
        let mut a: Zn<&mut [u8]> = res.to_mut();
        let base2k: u64 = 1 << basek;
        let mask: u64 = base2k - 1;
        let base2k_half: i64 = (base2k >> 1) as i64;
        (0..k.div_ceil(basek)).for_each(|j| {
            a.at_mut(res_col, j)[..n]
                .iter_mut()
                .for_each(|x| *x = (source.next_u64n(base2k, mask) as i64) - base2k_half);
        })
    }
}

unsafe impl ZnFillDistF64Impl<Self> for FFT64 {
    fn zn_fill_dist_f64_impl<R, D: rand::prelude::Distribution<f64>>(
        n: usize,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) where
        R: ZnToMut,
    {
        let mut a: Zn<&mut [u8]> = res.to_mut();
        assert!(
            (bound.log2().ceil() as i64) < 64,
            "invalid bound: ceil(log2(bound))={} > 63",
            (bound.log2().ceil() as i64)
        );

        let limb: usize = k.div_ceil(basek) - 1;
        let basek_rem: usize = (limb + 1) * basek - k;

        if basek_rem != 0 {
            a.at_mut(res_col, limb)[..n].iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a = (dist_f64.round() as i64) << basek_rem;
            });
        } else {
            a.at_mut(res_col, limb)[..n].iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a = dist_f64.round() as i64
            });
        }
    }
}

unsafe impl ZnAddDistF64Impl<Self> for FFT64 {
    fn zn_add_dist_f64_impl<R, D: rand::prelude::Distribution<f64>>(
        n: usize,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) where
        R: ZnToMut,
    {
        let mut a: Zn<&mut [u8]> = res.to_mut();
        assert!(
            (bound.log2().ceil() as i64) < 64,
            "invalid bound: ceil(log2(bound))={} > 63",
            (bound.log2().ceil() as i64)
        );

        let limb: usize = k.div_ceil(basek) - 1;
        let basek_rem: usize = (limb + 1) * basek - k;

        if basek_rem != 0 {
            a.at_mut(res_col, limb)[..n].iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a += (dist_f64.round() as i64) << basek_rem;
            });
        } else {
            a.at_mut(res_col, limb)[..n].iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a += dist_f64.round() as i64
            });
        }
    }
}

unsafe impl ZnFillNormalImpl<Self> for FFT64
where
    Self: ZnFillDistF64Impl<Self>,
{
    fn zn_fill_normal_impl<R>(
        n: usize,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: ZnToMut,
    {
        Self::zn_fill_dist_f64_impl(
            n,
            basek,
            res,
            res_col,
            k,
            source,
            Normal::new(0.0, sigma).unwrap(),
            bound,
        );
    }
}

unsafe impl ZnAddNormalImpl<Self> for FFT64
where
    Self: ZnAddDistF64Impl<Self>,
{
    fn zn_add_normal_impl<R>(
        n: usize,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: ZnToMut,
    {
        Self::zn_add_dist_f64_impl(
            n,
            basek,
            res,
            res_col,
            k,
            source,
            Normal::new(0.0, sigma).unwrap(),
            bound,
        );
    }
}
