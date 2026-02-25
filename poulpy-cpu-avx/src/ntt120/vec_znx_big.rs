//! Large-coefficient (i128) ring element vector operations for [`NTT120Avx`](super::NTT120Avx).
//!
//! Implements the `VecZnxBig*` OEP traits. `VecZnxBig` stores ring element vectors
//! using `ScalarBig = i128` (one i128 per coefficient), enabling exact CRT accumulation
//! of NTT-domain products before normalization back to the base-2^k representation.
//!
//! All implementations delegate to the `ntt120_vec_znx_big_*` reference functions in
//! `poulpy_hal::reference::ntt120::vec_znx_big`.

use super::NTT120Avx;
use poulpy_hal::{
    api::{TakeSlice, VecZnxBigAutomorphismInplaceTmpBytes, VecZnxBigNormalizeTmpBytes},
    layouts::{Module, Scratch, VecZnxBigToMut, VecZnxBigToRef, VecZnxToMut, VecZnxToRef},
    oep::{
        TakeSliceImpl, VecZnxBigAddImpl, VecZnxBigAddInplaceImpl, VecZnxBigAddNormalImpl, VecZnxBigAddSmallImpl,
        VecZnxBigAddSmallInplaceImpl, VecZnxBigAutomorphismImpl, VecZnxBigAutomorphismInplaceImpl,
        VecZnxBigAutomorphismInplaceTmpBytesImpl, VecZnxBigFromSmallImpl, VecZnxBigNegateImpl, VecZnxBigNegateInplaceImpl,
        VecZnxBigNormalizeImpl, VecZnxBigNormalizeTmpBytesImpl, VecZnxBigSubImpl, VecZnxBigSubInplaceImpl,
        VecZnxBigSubNegateInplaceImpl, VecZnxBigSubSmallAImpl, VecZnxBigSubSmallBImpl, VecZnxBigSubSmallInplaceImpl,
        VecZnxBigSubSmallNegateInplaceImpl,
    },
    reference::ntt120::vec_znx_big::{
        ntt120_vec_znx_big_add, ntt120_vec_znx_big_add_inplace, ntt120_vec_znx_big_add_normal_ref, ntt120_vec_znx_big_add_small,
        ntt120_vec_znx_big_add_small_inplace, ntt120_vec_znx_big_automorphism, ntt120_vec_znx_big_automorphism_inplace,
        ntt120_vec_znx_big_automorphism_inplace_tmp_bytes, ntt120_vec_znx_big_from_small, ntt120_vec_znx_big_negate,
        ntt120_vec_znx_big_negate_inplace, ntt120_vec_znx_big_normalize, ntt120_vec_znx_big_normalize_tmp_bytes,
        ntt120_vec_znx_big_sub, ntt120_vec_znx_big_sub_inplace, ntt120_vec_znx_big_sub_negate_inplace,
        ntt120_vec_znx_big_sub_small_a, ntt120_vec_znx_big_sub_small_b, ntt120_vec_znx_big_sub_small_inplace,
        ntt120_vec_znx_big_sub_small_negate_inplace,
    },
    source::Source,
};

unsafe impl VecZnxBigFromSmallImpl<Self> for NTT120Avx {
    fn vec_znx_big_from_small_impl<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxToRef,
    {
        ntt120_vec_znx_big_from_small::<_, _, NTT120Avx>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxBigAddNormalImpl<Self> for NTT120Avx {
    fn add_normal_impl<R: VecZnxBigToMut<Self>>(
        _module: &Module<Self>,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) {
        ntt120_vec_znx_big_add_normal_ref::<_, NTT120Avx>(base2k, res, res_col, k, sigma, bound, source);
    }
}

unsafe impl VecZnxBigAddImpl<Self> for NTT120Avx {
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
        ntt120_vec_znx_big_add::<_, _, _, NTT120Avx>(res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl VecZnxBigAddInplaceImpl<Self> for NTT120Avx {
    fn vec_znx_big_add_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
    {
        ntt120_vec_znx_big_add_inplace::<_, _, NTT120Avx>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxBigAddSmallImpl<Self> for NTT120Avx {
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
        ntt120_vec_znx_big_add_small::<_, _, _, NTT120Avx>(res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl VecZnxBigAddSmallInplaceImpl<Self> for NTT120Avx {
    fn vec_znx_big_add_small_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxToRef,
    {
        ntt120_vec_znx_big_add_small_inplace::<_, _, NTT120Avx>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxBigSubImpl<Self> for NTT120Avx {
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
        ntt120_vec_znx_big_sub::<_, _, _, NTT120Avx>(res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl VecZnxBigSubInplaceImpl<Self> for NTT120Avx {
    fn vec_znx_big_sub_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
    {
        ntt120_vec_znx_big_sub_inplace::<_, _, NTT120Avx>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxBigSubNegateInplaceImpl<Self> for NTT120Avx {
    fn vec_znx_big_sub_negate_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
    {
        ntt120_vec_znx_big_sub_negate_inplace::<_, _, NTT120Avx>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxBigSubSmallAImpl<Self> for NTT120Avx {
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
        ntt120_vec_znx_big_sub_small_a::<_, _, _, NTT120Avx>(res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl VecZnxBigSubSmallInplaceImpl<Self> for NTT120Avx {
    fn vec_znx_big_sub_small_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxToRef,
    {
        ntt120_vec_znx_big_sub_small_inplace::<_, _, NTT120Avx>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxBigSubSmallBImpl<Self> for NTT120Avx {
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
        ntt120_vec_znx_big_sub_small_b::<_, _, _, NTT120Avx>(res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl VecZnxBigSubSmallNegateInplaceImpl<Self> for NTT120Avx {
    fn vec_znx_big_sub_small_negate_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxToRef,
    {
        ntt120_vec_znx_big_sub_small_negate_inplace::<_, _, NTT120Avx>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxBigNegateImpl<Self> for NTT120Avx {
    fn vec_znx_big_negate_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
    {
        ntt120_vec_znx_big_negate::<_, _, NTT120Avx>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxBigNegateInplaceImpl<Self> for NTT120Avx {
    fn vec_znx_big_negate_inplace_impl<R>(_module: &Module<Self>, res: &mut R, res_col: usize)
    where
        R: VecZnxBigToMut<Self>,
    {
        ntt120_vec_znx_big_negate_inplace::<_, NTT120Avx>(res, res_col);
    }
}

unsafe impl VecZnxBigNormalizeTmpBytesImpl<Self> for NTT120Avx {
    fn vec_znx_big_normalize_tmp_bytes_impl(module: &Module<Self>) -> usize {
        ntt120_vec_znx_big_normalize_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxBigNormalizeImpl<Self> for NTT120Avx
where
    Self: TakeSliceImpl<Self>,
{
    fn vec_znx_big_normalize_impl<R, A>(
        module: &Module<Self>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<Self>,
    {
        let (carry, _) = scratch.take_slice(module.vec_znx_big_normalize_tmp_bytes() / size_of::<i128>());
        ntt120_vec_znx_big_normalize::<_, _, NTT120Avx>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }
}

unsafe impl VecZnxBigAutomorphismImpl<Self> for NTT120Avx {
    fn vec_znx_big_automorphism_impl<R, A>(_module: &Module<Self>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxBigToRef<Self>,
    {
        ntt120_vec_znx_big_automorphism::<_, _, NTT120Avx>(p, res, res_col, a, a_col);
    }
}

unsafe impl VecZnxBigAutomorphismInplaceTmpBytesImpl<Self> for NTT120Avx {
    fn vec_znx_big_automorphism_inplace_tmp_bytes_impl(module: &Module<Self>) -> usize {
        ntt120_vec_znx_big_automorphism_inplace_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxBigAutomorphismInplaceImpl<Self> for NTT120Avx
where
    Module<Self>: VecZnxBigAutomorphismInplaceTmpBytes,
{
    fn vec_znx_big_automorphism_inplace_impl<R>(
        module: &Module<Self>,
        p: i64,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxBigToMut<Self>,
    {
        let (tmp, _) = scratch.take_slice(module.vec_znx_big_automorphism_inplace_tmp_bytes() / size_of::<i128>());
        ntt120_vec_znx_big_automorphism_inplace::<_, NTT120Avx>(p, res, res_col, tmp);
    }
}
