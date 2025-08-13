use std::fmt;

use rand_distr::{Distribution, Normal};
use sampling::source::Source;

use crate::{
    hal::{
        api::{
            TakeSlice, VecZnxBigAddDistF64, VecZnxBigFillDistF64, VecZnxBigNormalizeTmpBytes, ZnxInfos, ZnxSliceSize, ZnxView,
            ZnxViewMut,
        },
        layouts::{
            Data, DataRef, Module, Scratch, VecZnx, VecZnxBig, VecZnxBigBytesOf, VecZnxBigOwned, VecZnxBigToMut, VecZnxBigToRef,
            VecZnxToMut, VecZnxToRef,
        },
        oep::{
            VecZnxBigAddDistF64Impl, VecZnxBigAddImpl, VecZnxBigAddInplaceImpl, VecZnxBigAddNormalImpl, VecZnxBigAddSmallImpl,
            VecZnxBigAddSmallInplaceImpl, VecZnxBigAllocBytesImpl, VecZnxBigAllocImpl, VecZnxBigAutomorphismImpl,
            VecZnxBigAutomorphismInplaceImpl, VecZnxBigFillDistF64Impl, VecZnxBigFillNormalImpl, VecZnxBigFromBytesImpl,
            VecZnxBigNegateInplaceImpl, VecZnxBigNormalizeImpl, VecZnxBigNormalizeTmpBytesImpl, VecZnxBigSubABInplaceImpl,
            VecZnxBigSubBAInplaceImpl, VecZnxBigSubImpl, VecZnxBigSubSmallAImpl, VecZnxBigSubSmallAInplaceImpl,
            VecZnxBigSubSmallBImpl, VecZnxBigSubSmallBInplaceImpl,
        },
    },
    implementation::cpu_spqlios::{ffi::vec_znx, module_fft64::FFT64},
};

const VEC_ZNX_BIG_FFT64_WORDSIZE: usize = 1;

impl<D: DataRef> ZnxView for VecZnxBig<D, FFT64> {
    type Scalar = i64;
}

impl<D: Data> VecZnxBigBytesOf for VecZnxBig<D, FFT64> {
    fn bytes_of(n: usize, cols: usize, size: usize) -> usize {
        VEC_ZNX_BIG_FFT64_WORDSIZE * n * cols * size * size_of::<f64>()
    }
}

impl<D: Data> ZnxSliceSize for VecZnxBig<D, FFT64> {
    fn sl(&self) -> usize {
        VEC_ZNX_BIG_FFT64_WORDSIZE * self.n() * self.cols()
    }
}

unsafe impl VecZnxBigAllocImpl<FFT64> for FFT64 {
    fn vec_znx_big_alloc_impl(n: usize, cols: usize, size: usize) -> VecZnxBigOwned<FFT64> {
        VecZnxBig::<Vec<u8>, FFT64>::new(n, cols, size)
    }
}

unsafe impl VecZnxBigFromBytesImpl<FFT64> for FFT64 {
    fn vec_znx_big_from_bytes_impl(n: usize, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<FFT64> {
        VecZnxBig::<Vec<u8>, FFT64>::new_from_bytes(n, cols, size, bytes)
    }
}

unsafe impl VecZnxBigAllocBytesImpl<FFT64> for FFT64 {
    fn vec_znx_big_alloc_bytes_impl(n: usize, cols: usize, size: usize) -> usize {
        VecZnxBig::<Vec<u8>, FFT64>::bytes_of(n, cols, size)
    }
}

unsafe impl VecZnxBigAddDistF64Impl<FFT64> for FFT64 {
    fn add_dist_f64_impl<R: VecZnxBigToMut<FFT64>, D: Distribution<f64>>(
        _module: &Module<FFT64>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) {
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();
        assert!(
            (bound.log2().ceil() as i64) < 64,
            "invalid bound: ceil(log2(bound))={} > 63",
            (bound.log2().ceil() as i64)
        );

        let limb: usize = k.div_ceil(basek) - 1;
        let basek_rem: usize = (limb + 1) * basek - k;

        if basek_rem != 0 {
            res.at_mut(res_col, limb).iter_mut().for_each(|x| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *x += (dist_f64.round() as i64) << basek_rem;
            });
        } else {
            res.at_mut(res_col, limb).iter_mut().for_each(|x| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *x += dist_f64.round() as i64
            });
        }
    }
}

unsafe impl VecZnxBigAddNormalImpl<FFT64> for FFT64 {
    fn add_normal_impl<R: VecZnxBigToMut<FFT64>>(
        module: &Module<FFT64>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) {
        module.vec_znx_big_add_dist_f64(
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

unsafe impl VecZnxBigFillDistF64Impl<FFT64> for FFT64 {
    fn fill_dist_f64_impl<R: VecZnxBigToMut<FFT64>, D: Distribution<f64>>(
        _module: &Module<FFT64>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) {
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();
        assert!(
            (bound.log2().ceil() as i64) < 64,
            "invalid bound: ceil(log2(bound))={} > 63",
            (bound.log2().ceil() as i64)
        );

        let limb: usize = k.div_ceil(basek) - 1;
        let basek_rem: usize = (limb + 1) * basek - k;

        if basek_rem != 0 {
            res.at_mut(res_col, limb).iter_mut().for_each(|x| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *x = (dist_f64.round() as i64) << basek_rem;
            });
        } else {
            res.at_mut(res_col, limb).iter_mut().for_each(|x| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *x = dist_f64.round() as i64
            });
        }
    }
}

unsafe impl VecZnxBigFillNormalImpl<FFT64> for FFT64 {
    fn fill_normal_impl<R: VecZnxBigToMut<FFT64>>(
        module: &Module<FFT64>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) {
        module.vec_znx_big_fill_dist_f64(
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

unsafe impl VecZnxBigAddImpl<FFT64> for FFT64 {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_impl<R, A, B>(
        module: &Module<FFT64>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
    ) where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
        B: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let b: VecZnxBig<&[u8], FFT64> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
            assert_eq!(b.n(), res.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_add(
                module.ptr(),
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(b_col, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }
}

unsafe impl VecZnxBigAddInplaceImpl<FFT64> for FFT64 {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_inplace_impl<R, A>(module: &Module<FFT64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
        }
        unsafe {
            vec_znx::vec_znx_add(
                module.ptr(),
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
            )
        }
    }
}

unsafe impl VecZnxBigAddSmallImpl<FFT64> for FFT64 {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_small_impl<R, A, B>(
        module: &Module<FFT64>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
    ) where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
        B: VecZnxToRef,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let b: VecZnx<&[u8]> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
            assert_eq!(b.n(), res.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_add(
                module.ptr(),
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(b_col, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }
}

unsafe impl VecZnxBigAddSmallInplaceImpl<FFT64> for FFT64 {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_small_inplace_impl<R, A>(module: &Module<FFT64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
        }
        unsafe {
            vec_znx::vec_znx_add(
                module.ptr(),
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }
}

unsafe impl VecZnxBigSubImpl<FFT64> for FFT64 {
    /// Subtracts `a` to `b` and stores the result on `c`.
    fn vec_znx_big_sub_impl<R, A, B>(
        module: &Module<FFT64>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
    ) where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
        B: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let b: VecZnxBig<&[u8], FFT64> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
            assert_eq!(b.n(), res.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                module.ptr(),
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(b_col, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }
}

unsafe impl VecZnxBigSubABInplaceImpl<FFT64> for FFT64 {
    /// Subtracts `a` from `b` and stores the result on `b`.
    fn vec_znx_big_sub_ab_inplace_impl<R, A>(module: &Module<FFT64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                module.ptr(),
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }
}

unsafe impl VecZnxBigSubBAInplaceImpl<FFT64> for FFT64 {
    /// Subtracts `b` from `a` and stores the result on `b`.
    fn vec_znx_big_sub_ba_inplace_impl<R, A>(module: &Module<FFT64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                module.ptr(),
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
            )
        }
    }
}

unsafe impl VecZnxBigSubSmallAImpl<FFT64> for FFT64 {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_a_impl<R, A, B>(
        module: &Module<FFT64>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
    ) where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxToRef,
        B: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let b: VecZnxBig<&[u8], FFT64> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
            assert_eq!(b.n(), res.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                module.ptr(),
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(b_col, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }
}

unsafe impl VecZnxBigSubSmallAInplaceImpl<FFT64> for FFT64 {
    /// Subtracts `a` from `res` and stores the result on `res`.
    fn vec_znx_big_sub_small_a_inplace_impl<R, A>(module: &Module<FFT64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                module.ptr(),
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }
}

unsafe impl VecZnxBigSubSmallBImpl<FFT64> for FFT64 {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_b_impl<R, A, B>(
        module: &Module<FFT64>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
    ) where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
        B: VecZnxToRef,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let b: VecZnx<&[u8]> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
            assert_eq!(b.n(), res.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                module.ptr(),
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(b_col, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }
}

unsafe impl VecZnxBigSubSmallBInplaceImpl<FFT64> for FFT64 {
    /// Subtracts `res` from `a` and stores the result on `res`.
    fn vec_znx_big_sub_small_b_inplace_impl<R, A>(module: &Module<FFT64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                module.ptr(),
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
            )
        }
    }
}

unsafe impl VecZnxBigNegateInplaceImpl<FFT64> for FFT64 {
    fn vec_znx_big_negate_inplace_impl<A>(module: &Module<FFT64>, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<FFT64>,
    {
        let mut a: VecZnxBig<&mut [u8], FFT64> = a.to_mut();
        unsafe {
            vec_znx::vec_znx_negate(
                module.ptr(),
                a.at_mut_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }
}

unsafe impl VecZnxBigNormalizeTmpBytesImpl<FFT64> for FFT64 {
    fn vec_znx_big_normalize_tmp_bytes_impl(module: &Module<FFT64>, n: usize) -> usize {
        unsafe { vec_znx::vec_znx_normalize_base2k_tmp_bytes(module.ptr(), n as u64) as usize }
    }
}

unsafe impl VecZnxBigNormalizeImpl<FFT64> for FFT64 {
    fn vec_znx_big_normalize_impl<R, A>(
        module: &Module<FFT64>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<FFT64>,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), a.n());
        }

        let (tmp_bytes, _) = scratch.take_slice(module.vec_znx_big_normalize_tmp_bytes(a.n()));
        unsafe {
            vec_znx::vec_znx_normalize_base2k(
                module.ptr(),
                a.n() as u64,
                basek as u64,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }
}

unsafe impl VecZnxBigAutomorphismImpl<FFT64> for FFT64 {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `b`.
    fn vec_znx_big_automorphism_impl<R, A>(module: &Module<FFT64>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                module.ptr(),
                k,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }
}

unsafe impl VecZnxBigAutomorphismInplaceImpl<FFT64> for FFT64 {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `a`.
    fn vec_znx_big_automorphism_inplace_impl<A>(module: &Module<FFT64>, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<FFT64>,
    {
        let mut a: VecZnxBig<&mut [u8], FFT64> = a.to_mut();
        unsafe {
            vec_znx::vec_znx_automorphism(
                module.ptr(),
                k,
                a.at_mut_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }
}

impl<D: DataRef> fmt::Display for VecZnxBig<D, FFT64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "VecZnxBig(n={}, cols={}, size={})",
            self.n, self.cols, self.size
        )?;

        for col in 0..self.cols {
            writeln!(f, "Column {}:", col)?;
            for size in 0..self.size {
                let coeffs = self.at(col, size);
                write!(f, "  Size {}: [", size)?;

                let max_show = 100;
                let show_count = coeffs.len().min(max_show);

                for (i, &coeff) in coeffs.iter().take(show_count).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", coeff)?;
                }

                if coeffs.len() > max_show {
                    write!(f, ", ... ({} more)", coeffs.len() - max_show)?;
                }

                writeln!(f, "]")?;
            }
        }
        Ok(())
    }
}
