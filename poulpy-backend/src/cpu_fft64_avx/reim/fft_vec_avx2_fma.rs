/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn reim_add_avx2_fma(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    use std::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};

    let span: usize = res.len() >> 2;

    unsafe {
        let mut rr: *mut f64 = res.as_mut_ptr();
        let mut aa: *const f64 = a.as_ptr();
        let mut bb: *const f64 = b.as_ptr();

        for _ in (0..span).step_by(4) {
            let a_256: __m256d = _mm256_loadu_pd(aa);
            let b_256: __m256d = _mm256_loadu_pd(bb);
            _mm256_storeu_pd(rr, _mm256_add_pd(a_256, b_256));
            rr = rr.add(4);
            aa = aa.add(4);
            bb = bb.add(4);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn reim_add_inplace_avx2_fma(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    use std::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};

    let span: usize = res.len() >> 2;

    unsafe {
        let mut rr: *mut f64 = res.as_mut_ptr();
        let mut aa: *const f64 = a.as_ptr();

        for _ in (0..span).step_by(4) {
            let a_256: __m256d = _mm256_loadu_pd(aa);
            let r_256: __m256d = _mm256_loadu_pd(rr);
            _mm256_storeu_pd(rr, _mm256_add_pd(r_256, a_256));
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn reim_sub_avx2_fma(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    use std::arch::x86_64::{__m256d, _mm256_loadu_pd, _mm256_storeu_pd, _mm256_sub_pd};

    let span: usize = res.len() >> 2;

    unsafe {
        let mut rr: *mut f64 = res.as_mut_ptr();
        let mut aa: *const f64 = a.as_ptr();
        let mut bb: *const f64 = b.as_ptr();

        for _ in (0..span).step_by(4) {
            let a_256: __m256d = _mm256_loadu_pd(aa);
            let b_256: __m256d = _mm256_loadu_pd(bb);
            _mm256_storeu_pd(rr, _mm256_sub_pd(a_256, b_256));
            rr = rr.add(4);
            aa = aa.add(4);
            bb = bb.add(4);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn reim_sub_ab_inplace_avx2_fma(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    use std::arch::x86_64::{__m256d, _mm256_loadu_pd, _mm256_storeu_pd, _mm256_sub_pd};

    let span: usize = res.len() >> 2;

    unsafe {
        let mut rr: *mut f64 = res.as_mut_ptr();
        let mut aa: *const f64 = a.as_ptr();

        for _ in (0..span).step_by(4) {
            let a_256: __m256d = _mm256_loadu_pd(aa);
            let r_256: __m256d = _mm256_loadu_pd(rr);
            _mm256_storeu_pd(rr, _mm256_sub_pd(r_256, a_256));
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn reim_sub_ba_inplace_avx2_fma(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    use std::arch::x86_64::{__m256d, _mm256_loadu_pd, _mm256_storeu_pd, _mm256_sub_pd};

    let span: usize = res.len() >> 2;

    unsafe {
        let mut rr: *mut f64 = res.as_mut_ptr();
        let mut aa: *const f64 = a.as_ptr();

        for _ in (0..span).step_by(4) {
            let a_256: __m256d = _mm256_loadu_pd(aa);
            let r_256: __m256d = _mm256_loadu_pd(rr);
            _mm256_storeu_pd(rr, _mm256_sub_pd(a_256, r_256));
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn reim_negate_avx2_fma(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    use std::arch::x86_64::{__m256d, _mm256_loadu_pd, _mm256_storeu_pd, _mm256_xor_pd};

    let span: usize = res.len() >> 2;

    unsafe {
        use std::arch::x86_64::_mm256_set1_pd;

        let mut rr: *mut f64 = res.as_mut_ptr();
        let mut aa: *const f64 = a.as_ptr();

        let neg0: __m256d = _mm256_set1_pd(-0.0);

        for _ in (0..span).step_by(4) {
            let a_256: __m256d = _mm256_loadu_pd(aa);
            _mm256_storeu_pd(rr, _mm256_xor_pd(a_256, neg0));
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn reim_negate_inplace_avx2_fma(res: &mut [f64]) {
    use std::arch::x86_64::{__m256d, _mm256_loadu_pd, _mm256_storeu_pd, _mm256_xor_pd};

    let span: usize = res.len() >> 2;

    unsafe {
        use std::arch::x86_64::_mm256_set1_pd;

        let mut rr: *mut f64 = res.as_mut_ptr();
        let neg0: __m256d = _mm256_set1_pd(-0.0);

        for _ in (0..span).step_by(4) {
            let r_256: __m256d = _mm256_loadu_pd(rr);
            _mm256_storeu_pd(rr, _mm256_xor_pd(r_256, neg0));
            rr = rr.add(4);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn reim_addmul_avx2_fma(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    let m: usize = res.len() >> 1;

    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);
    let (br, bi) = b.split_at(m);

    unsafe {
        let mut rr_ptr: *mut f64 = rr.as_mut_ptr();
        let mut ri_ptr: *mut f64 = ri.as_mut_ptr();
        let mut ar_ptr: *const f64 = ar.as_ptr();
        let mut ai_ptr: *const f64 = ai.as_ptr();
        let mut br_ptr: *const f64 = br.as_ptr();
        let mut bi_ptr: *const f64 = bi.as_ptr();

        use std::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_loadu_pd, _mm256_storeu_pd};

        for _ in (0..m >> 2).step_by(4) {
            let mut rr: __m256d = _mm256_loadu_pd(rr_ptr);
            let mut ri: __m256d = _mm256_loadu_pd(ri_ptr);
            let ar: __m256d = _mm256_loadu_pd(ar_ptr);
            let ai: __m256d = _mm256_loadu_pd(ai_ptr);
            let br: __m256d = _mm256_loadu_pd(br_ptr);
            let bi: __m256d = _mm256_loadu_pd(bi_ptr);

            rr = _mm256_fmsub_pd(ai, bi, rr);
            rr = _mm256_fmsub_pd(ar, br, rr);
            ri = _mm256_fmadd_pd(ar, bi, ri);
            ri = _mm256_fmadd_pd(ai, br, ri);

            _mm256_storeu_pd(rr_ptr, rr);
            _mm256_storeu_pd(ri_ptr, ri);

            rr_ptr = rr_ptr.add(4);
            ri_ptr = ri_ptr.add(4);
            ar_ptr = ar_ptr.add(4);
            ai_ptr = ai_ptr.add(4);
            br_ptr = br_ptr.add(4);
            bi_ptr = bi_ptr.add(4);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn reim_mul_avx2_fma(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    let m: usize = res.len() >> 1;

    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);
    let (br, bi) = b.split_at(m);

    unsafe {
        let mut rr_ptr: *mut f64 = rr.as_mut_ptr();
        let mut ri_ptr: *mut f64 = ri.as_mut_ptr();
        let mut ar_ptr: *const f64 = ar.as_ptr();
        let mut ai_ptr: *const f64 = ai.as_ptr();
        let mut br_ptr: *const f64 = br.as_ptr();
        let mut bi_ptr: *const f64 = bi.as_ptr();

        use std::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_storeu_pd};

        for _ in (0..m >> 2).step_by(4) {
            let ar: __m256d = _mm256_loadu_pd(ar_ptr);
            let ai: __m256d = _mm256_loadu_pd(ai_ptr);
            let br: __m256d = _mm256_loadu_pd(br_ptr);
            let bi: __m256d = _mm256_loadu_pd(bi_ptr);

            let t1: __m256d = _mm256_mul_pd(ai, bi);
            let t2: __m256d = _mm256_mul_pd(ar, bi);

            let rr = _mm256_fmsub_pd(ar, br, t1);
            let ri = _mm256_fmadd_pd(ai, br, t2);

            _mm256_storeu_pd(rr_ptr, rr);
            _mm256_storeu_pd(ri_ptr, ri);

            rr_ptr = rr_ptr.add(4);
            ri_ptr = ri_ptr.add(4);
            ar_ptr = ar_ptr.add(4);
            ai_ptr = ai_ptr.add(4);
            br_ptr = br_ptr.add(4);
            bi_ptr = bi_ptr.add(4);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn reim_mul_inplace_avx2_fma(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    let m: usize = res.len() >> 1;

    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);

    unsafe {
        let mut rr_ptr: *mut f64 = rr.as_mut_ptr();
        let mut ri_ptr: *mut f64 = ri.as_mut_ptr();
        let mut ar_ptr: *const f64 = ar.as_ptr();
        let mut ai_ptr: *const f64 = ai.as_ptr();

        use std::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_storeu_pd};

        for _ in (0..m >> 2).step_by(4) {
            let ar: __m256d = _mm256_loadu_pd(ar_ptr);
            let ai: __m256d = _mm256_loadu_pd(ai_ptr);
            let br: __m256d = _mm256_loadu_pd(rr_ptr);
            let bi: __m256d = _mm256_loadu_pd(ri_ptr);

            let t1: __m256d = _mm256_mul_pd(ai, bi);
            let t2: __m256d = _mm256_mul_pd(ar, bi);

            let rr = _mm256_fmsub_pd(ar, br, t1);
            let ri = _mm256_fmadd_pd(ai, br, t2);

            _mm256_storeu_pd(rr_ptr, rr);
            _mm256_storeu_pd(ri_ptr, ri);

            rr_ptr = rr_ptr.add(4);
            ri_ptr = ri_ptr.add(4);
            ar_ptr = ar_ptr.add(4);
            ai_ptr = ai_ptr.add(4);
        }
    }
}
