//! Scalar-vector product for [`NTTIfma`](crate::NTTIfma).
//!
//! This module implements the SVP OEP traits on top of the IFMA NTT layout.
//! Preparation is shared with the reference IFMA pipeline, while the DFT-to-DFT
//! apply path uses a dedicated SIMD pointwise multiply kernel.

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m256i, __m512i, _mm256_loadu_si256, _mm256_madd52hi_epu64, _mm256_madd52lo_epu64, _mm256_setzero_si256,
    _mm256_storeu_si256, _mm512_loadu_si512, _mm512_madd52hi_epu64, _mm512_madd52lo_epu64, _mm512_setzero_si512,
    _mm512_storeu_si512,
};

use poulpy_hal::{
    layouts::{
        Module, ScalarZnxToRef, SvpPPol, SvpPPolToMut, SvpPPolToRef, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, ZnxInfos,
        ZnxView, ZnxViewMut,
    },
    oep::{SvpApplyDftToDftImpl, SvpApplyDftToDftInplaceImpl, SvpPrepareImpl},
    reference::ntt_ifma::svp::ntt_ifma_svp_prepare,
};

use super::mat_vec_ifma::{reduce_bbc_ifma_simd, reduce_bbc_ifma_simd_512};
use crate::NTTIfma;

/// AVX-512 pointwise multiply: processes 2 coefficients per iteration using `__m512i`.
///
/// Falls back to 256-bit for odd tail.
#[target_feature(enable = "avx512ifma")]
unsafe fn svp_pointwise_mul_ifma(n: usize, res: *mut __m256i, b: *const __m256i, a: *const __m256i) {
    unsafe {
        let res_512 = res as *mut __m512i;
        let b_512 = b as *const __m512i;
        let a_512 = a as *const __m512i;
        let zero_512 = _mm512_setzero_si512();

        // Main loop: 2 coefficients per iteration via __m512i
        let pairs = n / 2;
        for i in 0..pairs {
            let bv = _mm512_loadu_si512(b_512.add(i));
            let av = _mm512_loadu_si512(a_512.add(i));
            let acc_lo = _mm512_madd52lo_epu64(zero_512, bv, av);
            let acc_hi = _mm512_madd52hi_epu64(zero_512, bv, av);
            _mm512_storeu_si512(res_512.add(i), reduce_bbc_ifma_simd_512(acc_lo, acc_hi));
        }

        // Odd tail: 1 remaining coefficient via __m256i
        if !n.is_multiple_of(2) {
            let i = n - 1;
            let zero = _mm256_setzero_si256();
            let bv = _mm256_loadu_si256(b.add(i));
            let av = _mm256_loadu_si256(a.add(i));
            let acc_lo = _mm256_madd52lo_epu64(zero, bv, av);
            let acc_hi = _mm256_madd52hi_epu64(zero, bv, av);
            _mm256_storeu_si256(res.add(i), reduce_bbc_ifma_simd(acc_lo, acc_hi));
        }
    }
}

unsafe impl SvpPrepareImpl<Self> for NTTIfma {
    fn svp_prepare_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<Self>,
        A: ScalarZnxToRef,
    {
        ntt_ifma_svp_prepare::<R, A, Self>(module, res, res_col, a, a_col);
    }
}

unsafe impl SvpApplyDftToDftImpl<Self> for NTTIfma {
    fn svp_apply_dft_to_dft_impl<R, A, C>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        R: VecZnxDftToMut<Self>,
        A: SvpPPolToRef<Self>,
        C: VecZnxDftToRef<Self>,
    {
        let mut res: VecZnxDft<&mut [u8], Self> = res.to_mut();
        let a: SvpPPol<&[u8], Self> = a.to_ref();
        let b: VecZnxDft<&[u8], Self> = b.to_ref();

        let n = res.n();
        let res_size = res.size();
        let b_size = b.size();
        let min_size = res_size.min(b_size);

        // a (SvpPPol) is in c-format: u32 view over u64 data
        let a_ptr = cast_slice::<_, u64>(a.at(a_col, 0)).as_ptr() as *const __m256i;

        for j in 0..min_size {
            let res_ptr = cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).as_mut_ptr() as *mut __m256i;
            let b_ptr = cast_slice::<_, u64>(b.at(b_col, j)).as_ptr() as *const __m256i;
            unsafe { svp_pointwise_mul_ifma(n, res_ptr, b_ptr, a_ptr) };
        }

        for j in min_size..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
        }
    }
}

unsafe impl SvpApplyDftToDftInplaceImpl for NTTIfma {
    fn svp_apply_dft_to_dft_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: SvpPPolToRef<Self>,
    {
        let mut res: VecZnxDft<&mut [u8], Self> = res.to_mut();
        let a: SvpPPol<&[u8], Self> = a.to_ref();

        let n = res.n();
        let res_size = res.size();

        let a_ptr = cast_slice::<_, u64>(a.at(a_col, 0)).as_ptr() as *const __m256i;

        for j in 0..res_size {
            let res_ptr = cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).as_mut_ptr() as *mut __m256i;
            // For in-place: res serves as both input and output.
            // svp_pointwise_mul_ifma reads b[i] then writes res[i] per coefficient,
            // so aliasing is safe (each coefficient is read before written).
            unsafe { svp_pointwise_mul_ifma(n, res_ptr, res_ptr, a_ptr) };
        }
    }
}
