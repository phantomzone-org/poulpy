#[inline(always)]
pub fn znx_negate_i64_ref(res: &mut [i64], src: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), src.len())
    }

    for i in 0..res.len() {
        res[i] = -src[i]
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn znx_negate_i64_avx(res: &mut [i64], src: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), src.len())
    }

    let n: usize = res.len();

    use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_setzero_si256, _mm256_storeu_si256, _mm256_sub_epi64};
    let span: usize = n >> 2;

    unsafe {
        let mut aa: *const __m256i = src.as_ptr() as *const __m256i;
        let mut rr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
        let zero: __m256i = _mm256_setzero_si256();
        for _ in 0..span {
            let v: __m256i = _mm256_loadu_si256(aa);
            let neg: __m256i = _mm256_sub_epi64(zero, v);
            _mm256_storeu_si256(rr, neg);
            aa = aa.add(1);
            rr = rr.add(1);
        }
    }

    if !res.len().is_multiple_of(4) {
        znx_negate_i64_ref(&mut res[span << 2..], &src[span << 2..])
    }
}
