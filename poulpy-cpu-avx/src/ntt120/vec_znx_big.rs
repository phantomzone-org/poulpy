//! Large-coefficient (i128) ring element vector operations for [`NTT120Avx`](super::NTT120Avx).
//!
//! Implements the `VecZnxBig*` OEP traits. `VecZnxBig` stores ring element vectors
//! using `ScalarBig = i128` (one i128 per coefficient), enabling exact CRT accumulation
//! of NTT-domain products before normalization back to the base-2^k representation.
//!
//! All implementations delegate to the `ntt120_vec_znx_big_*` reference functions in
//! `poulpy_hal::reference::ntt120::vec_znx_big`.

use super::{
    NTT120Avx,
    vec_znx_big_avx::{
        nfc_final_step_inplace_avx2, nfc_final_step_inplace_scalar, nfc_middle_step_avx2, nfc_middle_step_inplace_avx2,
        nfc_middle_step_inplace_scalar, nfc_middle_step_scalar, vi128_add_avx2, vi128_add_inplace_avx2, vi128_add_small_avx2,
        vi128_add_small_inplace_avx2, vi128_from_small_avx2, vi128_neg_from_small_avx2, vi128_negate_avx2,
        vi128_negate_inplace_avx2, vi128_sub_avx2, vi128_sub_inplace_avx2, vi128_sub_negate_inplace_avx2, vi128_sub_small_a_avx2,
        vi128_sub_small_b_avx2, vi128_sub_small_inplace_avx2, vi128_sub_small_negate_inplace_avx2,
    },
};
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
    reference::ntt120::{I128BigOps, I128NormalizeOps},
    source::Source,
};

impl I128BigOps for NTT120Avx {
    #[inline(always)]
    fn i128_add(res: &mut [i128], a: &[i128], b: &[i128]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { vi128_add_avx2(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_add_inplace(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_add_inplace_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_add_small(res: &mut [i128], a: &[i128], b: &[i64]) {
        unsafe { vi128_add_small_avx2(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_add_small_inplace(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_add_small_inplace_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub(res: &mut [i128], a: &[i128], b: &[i128]) {
        unsafe { vi128_sub_avx2(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_inplace(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_sub_inplace_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_negate_inplace(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_sub_negate_inplace_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_small_a(res: &mut [i128], a: &[i64], b: &[i128]) {
        unsafe { vi128_sub_small_a_avx2(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_small_b(res: &mut [i128], a: &[i128], b: &[i64]) {
        unsafe { vi128_sub_small_b_avx2(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_small_inplace(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_sub_small_inplace_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_small_negate_inplace(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_sub_small_negate_inplace_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_negate(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_negate_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_negate_inplace(res: &mut [i128]) {
        unsafe { vi128_negate_inplace_avx2(res.len(), res) }
    }
    #[inline(always)]
    fn i128_neg_from_small(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_neg_from_small_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_from_small(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_from_small_avx2(res.len(), res, a) }
    }
}

impl I128NormalizeOps for NTT120Avx {
    #[inline(always)]
    fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        if base2k <= 64 && res.len() >= 4 {
            unsafe { nfc_middle_step_avx2(base2k as u32, lsh as u32, res.len(), res, a, carry) }
        } else {
            nfc_middle_step_scalar(base2k, lsh, res, a, carry);
        }
    }

    #[inline(always)]
    fn nfc_middle_step_inplace(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        if base2k <= 64 && res.len() >= 4 {
            unsafe { nfc_middle_step_inplace_avx2(base2k as u32, lsh as u32, res.len(), res, carry) }
        } else {
            nfc_middle_step_inplace_scalar(base2k, lsh, res, carry);
        }
    }

    #[inline(always)]
    fn nfc_final_step_inplace(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        if base2k <= 64 && res.len() >= 4 {
            unsafe { nfc_final_step_inplace_avx2(base2k as u32, lsh as u32, res.len(), res, carry) }
        } else {
            nfc_final_step_inplace_scalar(base2k, lsh, res, carry);
        }
    }
}

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
