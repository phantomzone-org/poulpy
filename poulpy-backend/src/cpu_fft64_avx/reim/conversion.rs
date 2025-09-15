/// # Correctness
/// Ensured for inputs absolute value bounded by 2^50-1
/// # Safety
/// Caller must ensure the CPU supports FMA (e.g., via `is_x86_feature_detected!("fma")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "fma")]
pub fn reim_from_znx_i64_bnd50_fma(res: &mut [f64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len())
    }

    let n: usize = res.len();

    unsafe {
        use std::arch::x86_64::{
            __m256d, __m256i, _mm256_add_epi64, _mm256_castsi256_pd, _mm256_loadu_si256, _mm256_or_pd, _mm256_set1_epi64x,
            _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
        };

        let expo: f64 = (1i64 << 52) as f64;
        let add_cst: i64 = 1i64 << 51;
        let sub_cst: f64 = (3i64 << 51) as f64;

        let expo_256: __m256d = _mm256_set1_pd(expo);
        let add_cst_256: __m256i = _mm256_set1_epi64x(add_cst);
        let sub_cst_256: __m256d = _mm256_set1_pd(sub_cst);

        let mut res_ptr: *mut f64 = res.as_mut_ptr();
        let mut a_ptr: *const __m256i = a.as_ptr() as *const __m256i;

        let span: usize = n >> 2;

        for _ in 0..span {
            let mut ai64_256: __m256i = _mm256_loadu_si256(a_ptr);

            ai64_256 = _mm256_add_epi64(ai64_256, add_cst_256);

            let mut af64_256: __m256d = _mm256_castsi256_pd(ai64_256);
            af64_256 = _mm256_or_pd(af64_256, expo_256);
            af64_256 = _mm256_sub_pd(af64_256, sub_cst_256);

            _mm256_storeu_pd(res_ptr, af64_256);

            res_ptr = res_ptr.add(4);
            a_ptr = a_ptr.add(1);
        }

        if !res.len().is_multiple_of(4) {
            use poulpy_hal::reference::fft64::reim::reim_from_znx_i64_ref;
            reim_from_znx_i64_ref(&mut res[span << 2..], &a[span << 2..])
        }
    }
}

/// # Correctness
/// Only ensured for inputs absoluate value bounded by 2^63-1
/// # Safety
/// Caller must ensure the CPU supports FMA (e.g., via `is_x86_feature_detected!("fma,avx2")`);
#[allow(dead_code)]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn reim_to_znx_i64_bnd63_avx2_fma(res: &mut [i64], divisor: f64, a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len())
    }

    let sign_mask: u64 = 0x8000000000000000u64;
    let expo_mask: u64 = 0x7FF0000000000000u64;
    let mantissa_mask: u64 = (i64::MAX as u64) ^ expo_mask;
    let mantissa_msb: u64 = 0x0010000000000000u64;
    let divi_bits: f64 = divisor * (1i64 << 52) as f64;
    let offset: f64 = divisor / 2.;

    unsafe {
        use std::arch::x86_64::{
            __m256d, __m256i, _mm256_add_pd, _mm256_and_pd, _mm256_and_si256, _mm256_castpd_si256, _mm256_castsi256_pd,
            _mm256_loadu_pd, _mm256_or_pd, _mm256_or_si256, _mm256_set1_epi64x, _mm256_set1_pd, _mm256_sllv_epi64,
            _mm256_srli_epi64, _mm256_srlv_epi64, _mm256_sub_epi64, _mm256_xor_si256,
        };

        let sign_mask_256: __m256d = _mm256_castsi256_pd(_mm256_set1_epi64x(sign_mask as i64));
        let expo_mask_256: __m256i = _mm256_set1_epi64x(expo_mask as i64);
        let mantissa_mask_256: __m256i = _mm256_set1_epi64x(mantissa_mask as i64);
        let mantissa_msb_256: __m256i = _mm256_set1_epi64x(mantissa_msb as i64);
        let offset_256 = _mm256_set1_pd(offset);
        let divi_bits_256 = _mm256_castpd_si256(_mm256_set1_pd(divi_bits));

        let mut res_ptr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
        let mut a_ptr: *const f64 = a.as_ptr();

        let span: usize = res.len() >> 2;

        for _ in 0..span {
            // read the next value
            use std::arch::x86_64::_mm256_storeu_si256;
            let mut a: __m256d = _mm256_loadu_pd(a_ptr);

            // a += sign(a) * m/2
            let asign: __m256d = _mm256_and_pd(a, sign_mask_256);
            a = _mm256_add_pd(a, _mm256_or_pd(asign, offset_256));

            // sign: either 0 or -1
            let mut sign_mask: __m256i = _mm256_castpd_si256(asign);
            sign_mask = _mm256_sub_epi64(_mm256_set1_epi64x(0), _mm256_srli_epi64(sign_mask, 63));

            // compute the exponents
            let a0exp: __m256i = _mm256_and_si256(_mm256_castpd_si256(a), expo_mask_256);
            let mut a0lsh: __m256i = _mm256_sub_epi64(a0exp, divi_bits_256);
            let mut a0rsh: __m256i = _mm256_sub_epi64(divi_bits_256, a0exp);
            a0lsh = _mm256_srli_epi64(a0lsh, 52);
            a0rsh = _mm256_srli_epi64(a0rsh, 52);

            // compute the new mantissa
            let mut a0pos: __m256i = _mm256_and_si256(_mm256_castpd_si256(a), mantissa_mask_256);
            a0pos = _mm256_or_si256(a0pos, mantissa_msb_256);
            a0lsh = _mm256_sllv_epi64(a0pos, a0lsh);
            a0rsh = _mm256_srlv_epi64(a0pos, a0rsh);
            let mut out: __m256i = _mm256_or_si256(a0lsh, a0rsh);

            // negate if the sign was negative
            out = _mm256_xor_si256(out, sign_mask);
            out = _mm256_sub_epi64(out, sign_mask);

            // stores
            _mm256_storeu_si256(res_ptr, out);

            res_ptr = res_ptr.add(1);
            a_ptr = a_ptr.add(4);
        }

        if !res.len().is_multiple_of(4) {
            use poulpy_hal::reference::fft64::reim::reim_to_znx_i64_ref;
            reim_to_znx_i64_ref(&mut res[span << 2..], divisor, &a[span << 2..])
        }
    }
}

/// # Correctness
/// Only ensured for inputs absoluate value bounded by 2^63-1
/// # Safety
/// Caller must ensure the CPU supports FMA (e.g., via `is_x86_feature_detected!("fma,avx2")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn reim_to_znx_i64_inplace_bnd63_avx2_fma(res: &mut [f64], divisor: f64) {
    let sign_mask: u64 = 0x8000000000000000u64;
    let expo_mask: u64 = 0x7FF0000000000000u64;
    let mantissa_mask: u64 = (i64::MAX as u64) ^ expo_mask;
    let mantissa_msb: u64 = 0x0010000000000000u64;
    let divi_bits: f64 = divisor * (1i64 << 52) as f64;
    let offset: f64 = divisor / 2.;

    unsafe {
        use std::arch::x86_64::{
            __m256d, __m256i, _mm256_add_pd, _mm256_and_pd, _mm256_and_si256, _mm256_castpd_si256, _mm256_castsi256_pd,
            _mm256_loadu_pd, _mm256_or_pd, _mm256_or_si256, _mm256_set1_epi64x, _mm256_set1_pd, _mm256_sllv_epi64,
            _mm256_srli_epi64, _mm256_srlv_epi64, _mm256_sub_epi64, _mm256_xor_si256,
        };

        use poulpy_hal::reference::fft64::reim::reim_to_znx_i64_inplace_ref;

        let sign_mask_256: __m256d = _mm256_castsi256_pd(_mm256_set1_epi64x(sign_mask as i64));
        let expo_mask_256: __m256i = _mm256_set1_epi64x(expo_mask as i64);
        let mantissa_mask_256: __m256i = _mm256_set1_epi64x(mantissa_mask as i64);
        let mantissa_msb_256: __m256i = _mm256_set1_epi64x(mantissa_msb as i64);
        let offset_256: __m256d = _mm256_set1_pd(offset);
        let divi_bits_256: __m256i = _mm256_castpd_si256(_mm256_set1_pd(divi_bits));

        let mut res_ptr_4xi64: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
        let mut res_ptr_1xf64: *mut f64 = res.as_mut_ptr();

        let span: usize = res.len() >> 2;

        for _ in 0..span {
            // read the next value
            use std::arch::x86_64::_mm256_storeu_si256;
            let mut a: __m256d = _mm256_loadu_pd(res_ptr_1xf64);

            // a += sign(a) * m/2
            let asign: __m256d = _mm256_and_pd(a, sign_mask_256);
            a = _mm256_add_pd(a, _mm256_or_pd(asign, offset_256));

            // sign: either 0 or -1
            let mut sign_mask: __m256i = _mm256_castpd_si256(asign);
            sign_mask = _mm256_sub_epi64(_mm256_set1_epi64x(0), _mm256_srli_epi64(sign_mask, 63));

            // compute the exponents
            let a0exp: __m256i = _mm256_and_si256(_mm256_castpd_si256(a), expo_mask_256);
            let mut a0lsh: __m256i = _mm256_sub_epi64(a0exp, divi_bits_256);
            let mut a0rsh: __m256i = _mm256_sub_epi64(divi_bits_256, a0exp);
            a0lsh = _mm256_srli_epi64(a0lsh, 52);
            a0rsh = _mm256_srli_epi64(a0rsh, 52);

            // compute the new mantissa
            let mut a0pos: __m256i = _mm256_and_si256(_mm256_castpd_si256(a), mantissa_mask_256);
            a0pos = _mm256_or_si256(a0pos, mantissa_msb_256);
            a0lsh = _mm256_sllv_epi64(a0pos, a0lsh);
            a0rsh = _mm256_srlv_epi64(a0pos, a0rsh);
            let mut out: __m256i = _mm256_or_si256(a0lsh, a0rsh);

            // negate if the sign was negative
            out = _mm256_xor_si256(out, sign_mask);
            out = _mm256_sub_epi64(out, sign_mask);

            // stores
            _mm256_storeu_si256(res_ptr_4xi64, out);

            res_ptr_4xi64 = res_ptr_4xi64.add(1);
            res_ptr_1xf64 = res_ptr_1xf64.add(4);
        }

        if !res.len().is_multiple_of(4) {
            reim_to_znx_i64_inplace_ref(&mut res[span << 2..], divisor)
        }
    }
}

/// # Correctness
/// Only ensured for inputs absoluate value bounded by 2^50-1
/// # Safety
/// Caller must ensure the CPU supports FMA (e.g., via `is_x86_feature_detected!("fma")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "fma")]
#[allow(dead_code)]
pub fn reim_to_znx_i64_avx2_bnd50_fma(res: &mut [i64], divisor: f64, a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len())
    }

    unsafe {
        use std::arch::x86_64::{
            __m256d, __m256i, _mm256_add_pd, _mm256_and_si256, _mm256_castpd_si256, _mm256_loadu_pd, _mm256_set1_epi64x,
            _mm256_set1_pd, _mm256_storeu_si256, _mm256_sub_epi64,
        };

        let mantissa_mask: u64 = 0x000FFFFFFFFFFFFFu64;
        let sub_cst: i64 = 1i64 << 51;
        let add_cst: f64 = divisor * (3i64 << 51) as f64;

        let sub_cst_4: __m256i = _mm256_set1_epi64x(sub_cst);
        let add_cst_4: std::arch::x86_64::__m256d = _mm256_set1_pd(add_cst);
        let mantissa_mask_4: __m256i = _mm256_set1_epi64x(mantissa_mask as i64);

        let mut res_ptr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
        let mut a_ptr = a.as_ptr();

        let span: usize = res.len() >> 2;

        for _ in 0..span {
            // read the next value
            let mut a: __m256d = _mm256_loadu_pd(a_ptr);
            a = _mm256_add_pd(a, add_cst_4);
            let mut ai: __m256i = _mm256_castpd_si256(a);
            ai = _mm256_and_si256(ai, mantissa_mask_4);
            ai = _mm256_sub_epi64(ai, sub_cst_4);
            // store the next value
            _mm256_storeu_si256(res_ptr, ai);

            res_ptr = res_ptr.add(1);
            a_ptr = a_ptr.add(4);
        }

        if !res.len().is_multiple_of(4) {
            use poulpy_hal::reference::fft64::reim::reim_to_znx_i64_ref;
            reim_to_znx_i64_ref(&mut res[span << 2..], divisor, &a[span << 2..])
        }
    }
}
