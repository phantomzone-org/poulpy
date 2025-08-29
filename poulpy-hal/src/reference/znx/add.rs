pub fn znx_add_i64_ref(res: &mut [i64], a: &[i64], b: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
        assert_eq!(res.len(), b.len());
    }

    let n: usize = res.len();
    for i in 0..n {
        res[i] = a[i] + b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn znx_add_i64_avx(res: &mut [i64], a: &[i64], b: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
        assert_eq!(res.len(), b.len());
    }

    use core::arch::x86_64::{__m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_storeu_si256};

    let n: usize = res.len();

    let span: usize = n >> 2;

    let mut rr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
    let mut aa: *const __m256i = a.as_ptr() as *const __m256i;
    let mut bb: *const __m256i = b.as_ptr() as *const __m256i;

    unsafe {
        for _ in 0..span {
            let sum: __m256i = _mm256_add_epi64(_mm256_loadu_si256(aa), _mm256_loadu_si256(bb));
            _mm256_storeu_si256(rr, sum);
            rr = rr.add(1);
            aa = aa.add(1);
            bb = bb.add(1);
        }
    }

    // tail
    if !res.len().is_multiple_of(4) {
        znx_add_i64_ref(&mut res[span << 2..], &a[span << 2..], &b[span << 2..]);
    }
}

pub fn znx_add_inplace_i64_ref(res: &mut [i64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
    }

    let n: usize = res.len();
    for i in 0..n {
        res[i] += a[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn znx_add_inplace_i64_avx(res: &mut [i64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
    }

    use core::arch::x86_64::{__m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_storeu_si256};

    let n: usize = res.len();

    let span: usize = n >> 2;

    let mut rr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
    let mut aa: *const __m256i = a.as_ptr() as *const __m256i;

    unsafe {
        for _ in 0..span {
            let sum: __m256i = _mm256_add_epi64(_mm256_loadu_si256(rr), _mm256_loadu_si256(aa));
            _mm256_storeu_si256(rr, sum);
            rr = rr.add(1);
            aa = aa.add(1);
        }
    }

    // tail
    if !res.len().is_multiple_of(4) {
        znx_add_inplace_i64_ref(&mut res[span << 2..], &a[span << 2..]);
    }
}
