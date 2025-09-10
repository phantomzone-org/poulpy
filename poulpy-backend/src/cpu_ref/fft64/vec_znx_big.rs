use crate::cpu_ref::FFT64;
use poulpy_hal::{
    api::{TakeSlice, VecZnxBigAutomorphismInplaceTmpBytes, VecZnxBigNormalizeTmpBytes},
    layouts::{Backend, Module, Scratch, VecZnxBig, VecZnxBigOwned, VecZnxBigToMut, VecZnxBigToRef, VecZnxToMut, VecZnxToRef},
    oep::{
        TakeSliceImpl, VecZnxBigAddImpl, VecZnxBigAddInplaceImpl, VecZnxBigAddNormalImpl, VecZnxBigAddSmallImpl,
        VecZnxBigAddSmallInplaceImpl, VecZnxBigAllocBytesImpl, VecZnxBigAllocImpl, VecZnxBigAutomorphismImpl,
        VecZnxBigAutomorphismInplaceImpl, VecZnxBigAutomorphismInplaceTmpBytesImpl, VecZnxBigFromBytesImpl, VecZnxBigNegateImpl,
        VecZnxBigNegateInplaceImpl, VecZnxBigNormalizeImpl, VecZnxBigNormalizeTmpBytesImpl, VecZnxBigSubABInplaceImpl,
        VecZnxBigSubBAInplaceImpl, VecZnxBigSubImpl, VecZnxBigSubSmallAImpl, VecZnxBigSubSmallAInplaceImpl,
        VecZnxBigSubSmallBImpl, VecZnxBigSubSmallBInplaceImpl,
    },
    reference::{
        vec_znx_big::{
            vec_znx_big_add, vec_znx_big_add_inplace, vec_znx_big_add_normal_ref, vec_znx_big_add_small,
            vec_znx_big_add_small_inplace, vec_znx_big_automorphism, vec_znx_big_automorphism_inplace,
            vec_znx_big_automorphism_inplace_tmp_bytes, vec_znx_big_negate, vec_znx_big_negate_inplace, vec_znx_big_normalize,
            vec_znx_big_normalize_tmp_bytes, vec_znx_big_sub, vec_znx_big_sub_ab_inplace, vec_znx_big_sub_ba_inplace,
            vec_znx_big_sub_small_a, vec_znx_big_sub_small_a_inplace, vec_znx_big_sub_small_b, vec_znx_big_sub_small_b_inplace,
        },
        znx::{ZnxArithmeticAvx, ZnxArithmeticRef, ZnxNormalizeAvx, ZnxNormalizeRef},
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

unsafe impl VecZnxBigAddNormalImpl<Self> for FFT64 {
    fn add_normal_impl<R: VecZnxBigToMut<Self>>(
        _module: &Module<Self>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) {
        vec_znx_big_add_normal_ref(basek, res, res_col, k, sigma, bound, source);
    }
}

unsafe impl VecZnxBigAddImpl<Self> for FFT64 {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_impl<R, A, B>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
    ) where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
        B: VecZnxBigToRef<Self>,
    {
        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_add::<_, _, _, _, ZnxArithmeticAvx>(res, res_col, a, a_col, b, b_col);
        } else {
            vec_znx_big_add::<_, _, _, _, ZnxArithmeticRef>(res, res_col, a, a_col, b, b_col);
        }
    }
}

unsafe impl VecZnxBigAddInplaceImpl<Self> for FFT64 {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
    {
        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_add_inplace::<_, _, _, ZnxArithmeticAvx>(res, res_col, a, a_col);
        } else {
            vec_znx_big_add_inplace::<_, _, _, ZnxArithmeticRef>(res, res_col, a, a_col);
        }
    }
}

unsafe impl VecZnxBigAddSmallImpl<Self> for FFT64 {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_small_impl<R, A, B>(
        _module: &Module<Self>,
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
        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_add_small::<_, _, _, _, ZnxArithmeticAvx>(res, res_col, a, a_col, b, b_col);
        } else {
            vec_znx_big_add_small::<_, _, _, _, ZnxArithmeticRef>(res, res_col, a, a_col, b, b_col);
        }
    }
}

unsafe impl VecZnxBigAddSmallInplaceImpl<Self> for FFT64 {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_small_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxToRef,
    {
        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_add_small_inplace::<_, _, _, ZnxArithmeticAvx>(res, res_col, a, a_col);
        } else {
            vec_znx_big_add_small_inplace::<_, _, _, ZnxArithmeticRef>(res, res_col, a, a_col);
        }
    }
}

unsafe impl VecZnxBigSubImpl<Self> for FFT64 {
    /// Subtracts `a` to `b` and stores the result on `c`.
    fn vec_znx_big_sub_impl<R, A, B>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
    ) where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
        B: VecZnxBigToRef<Self>,
    {
        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_sub::<_, _, _, _, ZnxArithmeticAvx>(res, res_col, a, a_col, b, b_col);
        } else {
            vec_znx_big_sub::<_, _, _, _, ZnxArithmeticRef>(res, res_col, a, a_col, b, b_col);
        }
    }
}

unsafe impl VecZnxBigSubABInplaceImpl<Self> for FFT64 {
    /// Subtracts `a` from `b` and stores the result on `b`.
    fn vec_znx_big_sub_ab_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
    {
        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_sub_ab_inplace::<_, _, _, ZnxArithmeticAvx>(res, res_col, a, a_col);
        } else {
            vec_znx_big_sub_ab_inplace::<_, _, _, ZnxArithmeticRef>(res, res_col, a, a_col);
        }
    }
}

unsafe impl VecZnxBigSubBAInplaceImpl<Self> for FFT64 {
    /// Subtracts `b` from `a` and stores the result on `b`.
    fn vec_znx_big_sub_ba_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
    {
        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_sub_ba_inplace::<_, _, _, ZnxArithmeticAvx>(res, res_col, a, a_col);
        } else {
            vec_znx_big_sub_ba_inplace::<_, _, _, ZnxArithmeticRef>(res, res_col, a, a_col);
        }
    }
}

unsafe impl VecZnxBigSubSmallAImpl<Self> for FFT64 {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_a_impl<R, A, B>(
        _module: &Module<Self>,
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
        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_sub_small_a::<_, _, _, _, ZnxArithmeticAvx>(res, res_col, a, a_col, b, b_col);
        } else {
            vec_znx_big_sub_small_a::<_, _, _, _, ZnxArithmeticRef>(res, res_col, a, a_col, b, b_col);
        }
    }
}

unsafe impl VecZnxBigSubSmallAInplaceImpl<Self> for FFT64 {
    /// Subtracts `a` from `res` and stores the result on `res`.
    fn vec_znx_big_sub_small_a_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxToRef,
    {
        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_sub_small_a_inplace::<_, _, _, ZnxArithmeticAvx>(res, res_col, a, a_col);
        } else {
            vec_znx_big_sub_small_a_inplace::<_, _, _, ZnxArithmeticRef>(res, res_col, a, a_col);
        }
    }
}

unsafe impl VecZnxBigSubSmallBImpl<Self> for FFT64 {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_b_impl<R, A, B>(
        _module: &Module<Self>,
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
        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_sub_small_b::<_, _, _, _, ZnxArithmeticAvx>(res, res_col, a, a_col, b, b_col);
        } else {
            vec_znx_big_sub_small_b::<_, _, _, _, ZnxArithmeticRef>(res, res_col, a, a_col, b, b_col);
        }
    }
}

unsafe impl VecZnxBigSubSmallBInplaceImpl<Self> for FFT64 {
    /// Subtracts `res` from `a` and stores the result on `res`.
    fn vec_znx_big_sub_small_b_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxToRef,
    {
        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_sub_small_b_inplace::<_, _, _, ZnxArithmeticAvx>(res, res_col, a, a_col);
        } else {
            vec_znx_big_sub_small_b_inplace::<_, _, _, ZnxArithmeticRef>(res, res_col, a, a_col);
        }
    }
}

unsafe impl VecZnxBigNegateImpl<Self> for FFT64 {
    fn vec_znx_big_negate_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
    {
        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_negate::<_, _, _, ZnxArithmeticAvx>(res, res_col, a, a_col);
        } else {
            vec_znx_big_negate::<_, _, _, ZnxArithmeticRef>(res, res_col, a, a_col);
        }
    }
}

unsafe impl VecZnxBigNegateInplaceImpl<Self> for FFT64 {
    fn vec_znx_big_negate_inplace_impl<R>(_module: &Module<Self>, res: &mut R, res_col: usize)
    where
        R: VecZnxBigToMut<Self>,
    {
        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_negate_inplace::<_, _, ZnxArithmeticAvx>(res, res_col);
        } else {
            vec_znx_big_negate_inplace::<_, _, ZnxArithmeticRef>(res, res_col);
        }
    }
}

unsafe impl VecZnxBigNormalizeTmpBytesImpl<Self> for FFT64 {
    fn vec_znx_big_normalize_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_big_normalize_tmp_bytes(module.n())
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
        let (carry, _) = scratch.take_slice(module.vec_znx_big_normalize_tmp_bytes() / size_of::<i64>());

        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_normalize::<_, _, _, ZnxArithmeticAvx, ZnxNormalizeAvx>(basek, res, res_col, a, a_col, carry);
        } else {
            vec_znx_big_normalize::<_, _, _, ZnxArithmeticRef, ZnxNormalizeRef>(basek, res, res_col, a, a_col, carry);
        }
    }
}

unsafe impl VecZnxBigAutomorphismImpl<Self> for FFT64 {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `b`.
    fn vec_znx_big_automorphism_impl<R, A>(_module: &Module<Self>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
    {
        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_automorphism::<_, _, _, ZnxArithmeticAvx>(p, res, res_col, a, a_col);
        } else {
            vec_znx_big_automorphism::<_, _, _, ZnxArithmeticRef>(p, res, res_col, a, a_col);
        }
    }
}

unsafe impl VecZnxBigAutomorphismInplaceTmpBytesImpl<Self> for FFT64 {
    fn vec_znx_big_automorphism_inplace_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_big_automorphism_inplace_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxBigAutomorphismInplaceImpl<Self> for FFT64
where
    Module<Self>: VecZnxBigAutomorphismInplaceTmpBytes,
{
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `a`.
    fn vec_znx_big_automorphism_inplace_impl<R>(
        module: &Module<Self>,
        p: i64,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxBigToMut<Self>,
    {
        let (tmp, _) = scratch.take_slice(module.vec_znx_big_normalize_tmp_bytes() / size_of::<i64>());

        if std::is_x86_feature_detected!("avx2") {
            vec_znx_big_automorphism_inplace::<_, _, ZnxArithmeticAvx>(p, res, res_col, tmp);
        } else {
            vec_znx_big_automorphism_inplace::<_, _, ZnxArithmeticRef>(p, res, res_col, tmp);
        }
    }
}
