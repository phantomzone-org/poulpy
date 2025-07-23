use crate::{
    Scratch, VecZnxBigAdd, VecZnxBigAddInplace, VecZnxBigAddSmall, VecZnxBigAddSmallInplace, VecZnxBigAutomorphism,
    VecZnxBigAutomorphismInplace, VecZnxBigBytesOf, VecZnxBigNegateInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
    VecZnxBigSub, VecZnxBigSubABInplace, VecZnxBigSubBAInplace, VecZnxBigSubSmallA, VecZnxBigSubSmallAInplace,
    VecZnxBigSubSmallB, VecZnxBigSubSmallBInplace, VecZnxToMut, VecZnxToRef, ZnxSliceSize, ZnxViewMut, ffi::vec_znx,
};
use std::fmt;

use rand_distr::{Distribution, Normal};
use sampling::source::Source;

const VEC_ZNX_BIG_FFT64_WORDSIZE: usize = 1;

use crate::{
    FFT64, Module, VecZnx, VecZnxBig, VecZnxBigAddDistF64, VecZnxBigAddNormal, VecZnxBigAlloc, VecZnxBigAllocBytes,
    VecZnxBigFillDistF64, VecZnxBigFillNormal, VecZnxBigFromBytes, VecZnxBigOwned, VecZnxBigToMut, VecZnxBigToRef, ZnxInfos,
    ZnxView,
};

impl<D: AsRef<[u8]>> ZnxView for VecZnxBig<D, FFT64> {
    type Scalar = i64;
}

impl<D: AsRef<[u8]>> VecZnxBigBytesOf for VecZnxBig<D, FFT64> {
    fn bytes_of(n: usize, cols: usize, size: usize) -> usize {
        VEC_ZNX_BIG_FFT64_WORDSIZE * n * cols * size * size_of::<f64>()
    }
}

impl<D: AsRef<[u8]>> ZnxSliceSize for VecZnxBig<D, FFT64> {
    fn sl(&self) -> usize {
        VEC_ZNX_BIG_FFT64_WORDSIZE * self.n() * self.cols()
    }
}

impl VecZnxBigAlloc<FFT64> for Module<FFT64> {
    fn vec_znx_big_alloc(&self, cols: usize, size: usize) -> VecZnxBigOwned<FFT64> {
        VecZnxBig::<Vec<u8>, FFT64>::new(self.n(), cols, size)
    }
}

impl VecZnxBigFromBytes<FFT64> for Module<FFT64> {
    fn vec_znx_big_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<FFT64> {
        VecZnxBig::<Vec<u8>, FFT64>::new_from_bytes(self.n(), cols, size, bytes)
    }
}

impl VecZnxBigAllocBytes for Module<FFT64> {
    fn vec_znx_big_alloc_bytes(&self, cols: usize, size: usize) -> usize {
        VecZnxBig::<Vec<u8>, FFT64>::bytes_of(self.n(), cols, size)
    }
}

impl VecZnxBigAddDistF64<FFT64> for Module<FFT64> {
    fn add_dist_f64<R: VecZnxBigToMut<FFT64>, D: Distribution<f64>>(
        &self,
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

        let limb: usize = (k + basek - 1) / basek - 1;
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

impl VecZnxBigAddNormal<FFT64> for Module<FFT64> {
    fn add_normal<R: VecZnxBigToMut<FFT64>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) {
        self.add_dist_f64(
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

impl VecZnxBigFillDistF64<FFT64> for Module<FFT64> {
    fn fill_dist_f64<R: VecZnxBigToMut<FFT64>, D: Distribution<f64>>(
        &self,
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

        let limb: usize = (k + basek - 1) / basek - 1;
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

impl VecZnxBigFillNormal<FFT64> for Module<FFT64> {
    fn fill_normal<R: VecZnxBigToMut<FFT64>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) {
        self.fill_dist_f64(
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

impl VecZnxBigAdd<FFT64> for Module<FFT64> {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
        B: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let b: VecZnxBig<&[u8], FFT64> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
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

impl VecZnxBigAddInplace<FFT64> for Module<FFT64> {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
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

impl VecZnxBigAddSmall<FFT64> for Module<FFT64> {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_small<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
        B: VecZnxToRef,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let b: VecZnx<&[u8]> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
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

impl VecZnxBigAddSmallInplace<FFT64> for Module<FFT64> {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_small_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
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

impl VecZnxBigSub<FFT64> for Module<FFT64> {
    /// Subtracts `a` to `b` and stores the result on `c`.
    fn vec_znx_big_sub<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
        B: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let b: VecZnxBig<&[u8], FFT64> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
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

impl VecZnxBigSubABInplace<FFT64> for Module<FFT64> {
    /// Subtracts `a` from `b` and stores the result on `b`.
    fn vec_znx_big_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
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

impl VecZnxBigSubBAInplace<FFT64> for Module<FFT64> {
    /// Subtracts `b` from `a` and stores the result on `b`.
    fn vec_znx_big_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
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

impl VecZnxBigSubSmallA<FFT64> for Module<FFT64> {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_a<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxToRef,
        B: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let b: VecZnxBig<&[u8], FFT64> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
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

impl VecZnxBigSubSmallAInplace<FFT64> for Module<FFT64> {
    /// Subtracts `a` from `res` and stores the result on `res`.
    fn vec_znx_big_sub_small_a_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
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

impl VecZnxBigSubSmallB<FFT64> for Module<FFT64> {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_b<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
        B: VecZnxToRef,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let b: VecZnx<&[u8]> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
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

impl VecZnxBigSubSmallBInplace<FFT64> for Module<FFT64> {
    /// Subtracts `res` from `a` and stores the result on `res`.
    fn vec_znx_big_sub_small_b_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
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

impl VecZnxBigNegateInplace<FFT64> for Module<FFT64> {
    fn vec_znx_big_negate_inplace<A>(&self, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<FFT64>,
    {
        let mut a: VecZnxBig<&mut [u8], FFT64> = a.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_negate(
                self.ptr,
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

impl VecZnxBigNormalizeTmpBytes for Module<FFT64> {
    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize {
        unsafe { vec_znx::vec_znx_normalize_base2k_tmp_bytes(self.ptr) as usize }
    }
}

impl VecZnxBigNormalize<FFT64> for Module<FFT64> {
    fn vec_znx_big_normalize<R, A>(&self, basek: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxToMut,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }

        let (tmp_bytes, _) = scratch.tmp_slice(self.vec_znx_big_normalize_tmp_bytes());
        unsafe {
            vec_znx::vec_znx_normalize_base2k(
                self.ptr,
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

impl VecZnxBigAutomorphism<FFT64> for Module<FFT64> {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `b`.
    fn vec_znx_big_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.ptr,
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

impl VecZnxBigAutomorphismInplace<FFT64> for Module<FFT64> {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `a`.
    fn vec_znx_big_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<FFT64>,
    {
        let mut a: VecZnxBig<&mut [u8], FFT64> = a.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.ptr,
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

impl<D: AsRef<[u8]>> fmt::Display for VecZnxBig<D, FFT64> {
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
