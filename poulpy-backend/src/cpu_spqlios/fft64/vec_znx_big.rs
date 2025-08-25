use rand_distr::{Distribution, Normal};

use crate::cpu_spqlios::{FFT64, ffi::vec_znx};
use poulpy_hal::{
    api::{TakeSlice, VecZnxBigAddDistF64, VecZnxBigFillDistF64, VecZnxBigNormalizeTmpBytes},
    layouts::{
        Backend, Module, Scratch, VecZnx, VecZnxBig, VecZnxBigOwned, VecZnxBigToMut, VecZnxBigToRef, VecZnxToMut, VecZnxToRef,
        ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut,
    },
    oep::{
        TakeSliceImpl, VecZnxBigAddDistF64Impl, VecZnxBigAddImpl, VecZnxBigAddInplaceImpl, VecZnxBigAddNormalImpl,
        VecZnxBigAddSmallImpl, VecZnxBigAddSmallInplaceImpl, VecZnxBigAllocBytesImpl, VecZnxBigAllocImpl,
        VecZnxBigAutomorphismImpl, VecZnxBigAutomorphismInplaceImpl, VecZnxBigFillDistF64Impl, VecZnxBigFillNormalImpl,
        VecZnxBigFromBytesImpl, VecZnxBigNegateInplaceImpl, VecZnxBigNormalizeImpl, VecZnxBigNormalizeTmpBytesImpl,
        VecZnxBigSubABInplaceImpl, VecZnxBigSubBAInplaceImpl, VecZnxBigSubImpl, VecZnxBigSubSmallAImpl,
        VecZnxBigSubSmallAInplaceImpl, VecZnxBigSubSmallBImpl, VecZnxBigSubSmallBInplaceImpl,
    },
    source::Source,
};

unsafe impl VecZnxBigAllocBytesImpl<Self> for FFT64 {
    fn vec_znx_big_alloc_bytes_impl(n: usize, cols: usize, size: usize) -> usize {
        Self::layout_big_word_count() * n * cols * size * size_of::<f64>()
    }
}

unsafe impl VecZnxBigAllocImpl<Self> for FFT64 {
    fn vec_znx_big_alloc_impl(n: usize, cols: usize, size: usize) -> VecZnxBigOwned<Self> {
        VecZnxBig::alloc(n, cols, size)
    }
}

unsafe impl VecZnxBigFromBytesImpl<Self> for FFT64 {
    fn vec_znx_big_from_bytes_impl(n: usize, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<Self> {
        VecZnxBig::from_bytes(n, cols, size, bytes)
    }
}

unsafe impl VecZnxBigAddDistF64Impl<Self> for FFT64 {
    fn add_dist_f64_impl<R: VecZnxBigToMut<Self>, D: Distribution<f64>>(
        _module: &Module<Self>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) {
        let mut res: VecZnxBig<&mut [u8], Self> = res.to_mut();
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

unsafe impl VecZnxBigAddNormalImpl<Self> for FFT64 {
    fn add_normal_impl<R: VecZnxBigToMut<Self>>(
        module: &Module<Self>,
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

unsafe impl VecZnxBigFillDistF64Impl<Self> for FFT64 {
    fn fill_dist_f64_impl<R: VecZnxBigToMut<Self>, D: Distribution<f64>>(
        _module: &Module<Self>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) {
        let mut res: VecZnxBig<&mut [u8], Self> = res.to_mut();
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

unsafe impl VecZnxBigFillNormalImpl<Self> for FFT64 {
    fn fill_normal_impl<R: VecZnxBigToMut<Self>>(
        module: &Module<Self>,
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

unsafe impl VecZnxBigAddImpl<Self> for FFT64 {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_impl<R, A, B>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
        B: VecZnxBigToRef<Self>,
    {
        let a: VecZnxBig<&[u8], Self> = a.to_ref();
        let b: VecZnxBig<&[u8], Self> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], Self> = res.to_mut();

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

unsafe impl VecZnxBigAddInplaceImpl<Self> for FFT64 {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
    {
        let a: VecZnxBig<&[u8], Self> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], Self> = res.to_mut();

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

unsafe impl VecZnxBigAddSmallImpl<Self> for FFT64 {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_small_impl<R, A, B>(
        module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
    ) where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
        B: VecZnxToRef,
    {
        let a: VecZnxBig<&[u8], Self> = a.to_ref();
        let b: VecZnx<&[u8]> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], Self> = res.to_mut();

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

unsafe impl VecZnxBigAddSmallInplaceImpl<Self> for FFT64 {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_small_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], Self> = res.to_mut();

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

unsafe impl VecZnxBigSubImpl<Self> for FFT64 {
    /// Subtracts `a` to `b` and stores the result on `c`.
    fn vec_znx_big_sub_impl<R, A, B>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
        B: VecZnxBigToRef<Self>,
    {
        let a: VecZnxBig<&[u8], Self> = a.to_ref();
        let b: VecZnxBig<&[u8], Self> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], Self> = res.to_mut();

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

unsafe impl VecZnxBigSubABInplaceImpl<Self> for FFT64 {
    /// Subtracts `a` from `b` and stores the result on `b`.
    fn vec_znx_big_sub_ab_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
    {
        let a: VecZnxBig<&[u8], Self> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], Self> = res.to_mut();

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

unsafe impl VecZnxBigSubBAInplaceImpl<Self> for FFT64 {
    /// Subtracts `b` from `a` and stores the result on `b`.
    fn vec_znx_big_sub_ba_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
    {
        let a: VecZnxBig<&[u8], Self> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], Self> = res.to_mut();

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

unsafe impl VecZnxBigSubSmallAImpl<Self> for FFT64 {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_a_impl<R, A, B>(
        module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
    ) where
        R: VecZnxBigToMut<Self>,
        A: VecZnxToRef,
        B: VecZnxBigToRef<Self>,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let b: VecZnxBig<&[u8], Self> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], Self> = res.to_mut();

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

unsafe impl VecZnxBigSubSmallAInplaceImpl<Self> for FFT64 {
    /// Subtracts `a` from `res` and stores the result on `res`.
    fn vec_znx_big_sub_small_a_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], Self> = res.to_mut();

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

unsafe impl VecZnxBigSubSmallBImpl<Self> for FFT64 {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_b_impl<R, A, B>(
        module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
    ) where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
        B: VecZnxToRef,
    {
        let a: VecZnxBig<&[u8], Self> = a.to_ref();
        let b: VecZnx<&[u8]> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], Self> = res.to_mut();

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

unsafe impl VecZnxBigSubSmallBInplaceImpl<Self> for FFT64 {
    /// Subtracts `res` from `a` and stores the result on `res`.
    fn vec_znx_big_sub_small_b_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], Self> = res.to_mut();

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

unsafe impl VecZnxBigNegateInplaceImpl<Self> for FFT64 {
    fn vec_znx_big_negate_inplace_impl<A>(module: &Module<Self>, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<Self>,
    {
        let mut a: VecZnxBig<&mut [u8], Self> = a.to_mut();
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

unsafe impl VecZnxBigNormalizeTmpBytesImpl<Self> for FFT64 {
    fn vec_znx_big_normalize_tmp_bytes_impl(module: &Module<Self>) -> usize {
        unsafe { vec_znx::vec_znx_normalize_base2k_tmp_bytes(module.ptr()) as usize }
    }
}

unsafe impl VecZnxBigNormalizeImpl<Self> for FFT64
where
    Self: TakeSliceImpl<Self>,
{
    fn vec_znx_big_normalize_impl<R, A>(
        module: &Module<Self>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<Self>,
    {
        let a: VecZnxBig<&[u8], Self> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), a.n());
        }

        let (tmp_bytes, _) = scratch.take_slice(module.vec_znx_big_normalize_tmp_bytes());
        unsafe {
            vec_znx::vec_znx_normalize_base2k(
                module.ptr(),
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

unsafe impl VecZnxBigAutomorphismImpl<Self> for FFT64 {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `b`.
    fn vec_znx_big_automorphism_impl<R, A>(module: &Module<Self>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
    {
        let a: VecZnxBig<&[u8], Self> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], Self> = res.to_mut();

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

unsafe impl VecZnxBigAutomorphismInplaceImpl<Self> for FFT64 {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `a`.
    fn vec_znx_big_automorphism_inplace_impl<A>(module: &Module<Self>, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<Self>,
    {
        let mut a: VecZnxBig<&mut [u8], Self> = a.to_mut();
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
