use std::arch::x86_64::{__m256i, _mm256_set1_epi64x, _mm256_storeu_si256};

use itertools::izip;

#[inline(always)]
pub(crate) fn get_digit(basek: usize, x: i64) -> i64 {
    (x << (i64::BITS - basek as u32)) >> (i64::BITS - basek as u32)
}

#[inline(always)]
pub(crate) fn get_carry(basek: usize, x: i64, digit: i64) -> i64 {
    (x - digit) >> basek
}

pub fn znx_normalize_carry_only_ref(basek: usize, x: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len())
    }

    x.iter().zip(carry.iter_mut()).for_each(|(x, c)| {
        *c = get_carry(basek, *x, get_digit(basek, *x));
    });
}

pub fn znx_normalize_inplace_beg_ref(basek: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len())
    }

    x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
        let digit: i64 = get_digit(basek, *x);
        *c = get_carry(basek, *x, digit);
        *x = digit;
    });
}

pub fn znx_normalize_beg_ref(basek: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert_eq!(a.len(), carry.len());
    }

    izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
        let digit: i64 = get_digit(basek, *a);
        *c = get_carry(basek, *a, digit);
        *x = digit;
    });
}

pub fn znx_normalize_inplace_mid_ref(basek: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len())
    }
    x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
        let digit: i64 = get_digit(basek, *x);
        let carry: i64 = get_carry(basek, *x, digit);
        let digit_plus_c: i64 = digit + *c;
        *x = get_digit(basek, digit_plus_c);
        *c = carry + get_carry(basek, digit_plus_c, *x);
    });
}

pub fn znx_normalize_mid_ref(basek: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert_eq!(a.len(), carry.len());
    }
    izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
        let digit: i64 = get_digit(basek, *a);
        let carry: i64 = get_carry(basek, *a, digit);
        let digit_plus_c: i64 = digit + *c;
        *x = get_digit(basek, digit_plus_c);
        *c = carry + get_carry(basek, digit_plus_c, *x);
    });
}

pub fn znx_normalize_inplace_end_ref(basek: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len())
    }
    x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
        *x = get_digit(basek, get_digit(basek, *x) + *c);
    });
}

pub fn znx_normalize_end_ref(basek: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len())
    }
    izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
        *x = get_digit(basek, get_digit(basek, *a) + *c);
    });
}

/// Vector forms of those constants (broadcast to all lanes)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn normalize_consts_avx(basek: usize) -> (__m256i, __m256i, __m256i, __m256i) {
    assert!((1..=63).contains(&basek));
    let mask_k: i64 = ((1u64 << basek) - 1) as i64; // 0..k-1 bits set
    let sign_k: i64 = (1u64 << (basek - 1)) as i64; // bit k-1
    let topmask: i64 = (!0u64 << (64 - basek)) as i64; // top k bits set
    let sh_k: __m256i = _mm256_set1_epi64x(basek as i64);
    (
        _mm256_set1_epi64x(mask_k),  // mask_k_vec
        _mm256_set1_epi64x(sign_k),  // sign_k_vec
        sh_k,                        // shift_k_vec
        _mm256_set1_epi64x(topmask), // topmask_vec
    )
}

/// AVX2 get_digit using masks (no arithmetic shift needed):
/// digit = ((x & mask_k) ^ sign_k) - sign_k
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn get_digit_avx(x: __m256i, mask_k: __m256i, sign_k: __m256i) -> __m256i {
    use std::arch::x86_64::{_mm256_and_si256, _mm256_sub_epi64, _mm256_xor_si256};
    let low: __m256i = _mm256_and_si256(x, mask_k);
    let t: __m256i = _mm256_xor_si256(low, sign_k);
    _mm256_sub_epi64(t, sign_k)
}

/// AVX2 get_carry using precomputed shift and topmask:
/// carry = (x - digit) >>_arith k
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn get_carry_avx(
    x: __m256i,
    digit: __m256i,
    basek: __m256i,    // _mm256_set1_epi64x(k)
    top_mask: __m256i, // (!0 << (64 - k)) broadcast
) -> __m256i {
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_cmpgt_epi64, _mm256_or_si256, _mm256_setzero_si256, _mm256_srlv_epi64, _mm256_sub_epi64,
    };
    let diff: __m256i = _mm256_sub_epi64(x, digit);
    let lsr: __m256i = _mm256_srlv_epi64(diff, basek); // logical >>
    let neg: __m256i = _mm256_cmpgt_epi64(_mm256_setzero_si256(), diff); // 0xFFFF.. where v<0
    let fill: __m256i = _mm256_and_si256(neg, top_mask); // top k bits if negative
    _mm256_or_si256(lsr, fill)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn znx_normalize_carry_only_avx(basek: usize, x: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
    }

    use std::arch::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    let (mask, sign, basek_vec, top_mask) = normalize_consts_avx(basek);

    unsafe {
        let mut xx: *const __m256i = x.as_ptr() as *const __m256i;
        let mut cc: *mut __m256i = carry.as_ptr() as *mut __m256i;

        for _ in 0..span {
            let xx_256: __m256i = _mm256_loadu_si256(xx);

            //  (x << (64 - basek)) >> (64 - basek)
            let digit_256: __m256i = get_digit_avx(xx_256, mask, sign);

            // (x - digit) >> basek
            let carry_256: __m256i = get_carry_avx(xx_256, digit_256, basek_vec, top_mask);

            _mm256_storeu_si256(cc, carry_256);

            xx = xx.add(1);
            cc = cc.add(1);
        }
    }

    // tail
    if !x.len().is_multiple_of(4) {
        znx_normalize_carry_only_ref(basek, &x[span << 2..], &mut carry[span << 2..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn znx_normalize_inplace_beg_avx(basek: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
    }

    use std::arch::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    let (mask, sign, basek_vec, top_mask) = normalize_consts_avx(basek);

    unsafe {
        let mut xx: *mut __m256i = x.as_mut_ptr() as *mut __m256i;
        let mut cc: *mut __m256i = carry.as_ptr() as *mut __m256i;

        for _ in 0..span {
            let xx_256: __m256i = _mm256_loadu_si256(xx);

            //  (x << (64 - basek)) >> (64 - basek)
            let digit_256: __m256i = get_digit_avx(xx_256, mask, sign);

            // (x - digit) >> basek
            let carry_256: __m256i = get_carry_avx(xx_256, digit_256, basek_vec, top_mask);

            _mm256_storeu_si256(xx, digit_256);
            _mm256_storeu_si256(cc, carry_256);

            xx = xx.add(1);
            cc = cc.add(1);
        }
    }

    // tail
    if !x.len().is_multiple_of(4) {
        znx_normalize_inplace_beg_ref(basek, &mut x[span << 2..], &mut carry[span << 2..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn znx_normalize_beg_avx(basek: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert_eq!(a.len(), carry.len());
    }

    use std::arch::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    let (mask, sign, basek_vec, top_mask) = normalize_consts_avx(basek);

    unsafe {
        let mut xx: *mut __m256i = x.as_mut_ptr() as *mut __m256i;
        let mut aa: *const __m256i = a.as_ptr() as *const __m256i;
        let mut cc: *mut __m256i = carry.as_ptr() as *mut __m256i;

        for _ in 0..span {
            let aa_256: __m256i = _mm256_loadu_si256(aa);

            //  (x << (64 - basek)) >> (64 - basek)
            let digit_256: __m256i = get_digit_avx(aa_256, mask, sign);

            // (x - digit) >> basek
            let carry_256: __m256i = get_carry_avx(aa_256, digit_256, basek_vec, top_mask);

            _mm256_storeu_si256(xx, digit_256);
            _mm256_storeu_si256(cc, carry_256);

            xx = xx.add(1);
            aa = aa.add(1);
            cc = cc.add(1);
        }
    }

    // tail
    if !x.len().is_multiple_of(4) {
        znx_normalize_beg_ref(
            basek,
            &mut x[span << 2..],
            &a[span << 2..],
            &mut carry[span << 2..],
        );
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
pub fn znx_normalize_inplace_mid_avx(basek: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
    }

    use std::arch::x86_64::{_mm256_add_epi64, _mm256_loadu_si256, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    let (mask, sign, basek_vec, top_mask) = normalize_consts_avx(basek);

    unsafe {
        let mut xx: *mut __m256i = x.as_mut_ptr() as *mut __m256i;
        let mut cc: *mut __m256i = carry.as_ptr() as *mut __m256i;

        for _ in 0..span {
            let xx_256: __m256i = _mm256_loadu_si256(xx);
            let cc_256: __m256i = _mm256_loadu_si256(cc);

            let d0: __m256i = get_digit_avx(xx_256, mask, sign);
            let c0: __m256i = get_carry_avx(xx_256, d0, basek_vec, top_mask);

            let s: __m256i = _mm256_add_epi64(d0, cc_256);
            let x1: __m256i = get_digit_avx(s, mask, sign);
            let c1: __m256i = get_carry_avx(s, x1, basek_vec, top_mask);
            let cout: __m256i = _mm256_add_epi64(c0, c1);

            _mm256_storeu_si256(xx, x1);
            _mm256_storeu_si256(cc, cout);

            xx = xx.add(1);
            cc = cc.add(1);
        }
    }

    if !x.len().is_multiple_of(4) {
        znx_normalize_inplace_mid_ref(basek, &mut x[span << 2..], &mut carry[span << 2..]);
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
pub fn znx_normalize_mid_avx(basek: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert_eq!(a.len(), carry.len());
    }

    use std::arch::x86_64::{_mm256_add_epi64, _mm256_loadu_si256, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    let (mask, sign, basek_vec, top_mask) = normalize_consts_avx(basek);

    unsafe {
        let mut xx: *mut __m256i = x.as_mut_ptr() as *mut __m256i;
        let mut aa: *const __m256i = a.as_ptr() as *const __m256i;
        let mut cc: *mut __m256i = carry.as_ptr() as *mut __m256i;

        for _ in 0..span {
            let aa_256: __m256i = _mm256_loadu_si256(aa);
            let cc_256: __m256i = _mm256_loadu_si256(cc);

            let d0: __m256i = get_digit_avx(aa_256, mask, sign);
            let c0: __m256i = get_carry_avx(aa_256, d0, basek_vec, top_mask);

            let s: __m256i = _mm256_add_epi64(d0, cc_256);
            let x1: __m256i = get_digit_avx(s, mask, sign);
            let c1: __m256i = get_carry_avx(s, x1, basek_vec, top_mask);
            let cout: __m256i = _mm256_add_epi64(c0, c1);

            _mm256_storeu_si256(xx, x1);
            _mm256_storeu_si256(cc, cout);

            xx = xx.add(1);
            aa = aa.add(1);
            cc = cc.add(1);
        }
    }

    if !x.len().is_multiple_of(4) {
        znx_normalize_mid_ref(
            basek,
            &mut x[span << 2..],
            &a[span << 2..],
            &mut carry[span << 2..],
        );
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
pub fn znx_normalize_inplace_end_avx(basek: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
    }

    use std::arch::x86_64::{_mm256_add_epi64, _mm256_loadu_si256, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    let (mask, sign, _, _) = normalize_consts_avx(basek);

    unsafe {
        let mut xx: *mut __m256i = x.as_mut_ptr() as *mut __m256i;
        let mut cc: *mut __m256i = carry.as_ptr() as *mut __m256i;

        for _ in 0..span {
            let xv: __m256i = _mm256_loadu_si256(xx);
            let cv: __m256i = _mm256_loadu_si256(cc);

            let d0: __m256i = get_digit_avx(xv, mask, sign);
            let s: __m256i = _mm256_add_epi64(d0, cv);
            let x1: __m256i = get_digit_avx(s, mask, sign);

            _mm256_storeu_si256(xx, x1);

            xx = xx.add(1);
            cc = cc.add(1);
        }
    }

    if !x.len().is_multiple_of(4) {
        znx_normalize_inplace_end_ref(basek, &mut x[span << 2..], &mut carry[span << 2..]);
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
pub fn znx_normalize_end_avx(basek: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert_eq!(a.len(), carry.len());
    }

    use std::arch::x86_64::{_mm256_add_epi64, _mm256_loadu_si256, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    let (mask, sign, _, _) = normalize_consts_avx(basek);

    unsafe {
        let mut xx: *mut __m256i = x.as_mut_ptr() as *mut __m256i;
        let mut aa: *mut __m256i = a.as_ptr() as *mut __m256i;
        let mut cc: *mut __m256i = carry.as_ptr() as *mut __m256i;

        for _ in 0..span {
            let av: __m256i = _mm256_loadu_si256(aa);
            let cv: __m256i = _mm256_loadu_si256(cc);

            let d0: __m256i = get_digit_avx(av, mask, sign);
            let s: __m256i = _mm256_add_epi64(d0, cv);
            let x1: __m256i = get_digit_avx(s, mask, sign);

            _mm256_storeu_si256(xx, x1);

            xx = xx.add(1);
            aa = aa.add(1);
            cc = cc.add(1);
        }
    }

    if !x.len().is_multiple_of(4) {
        znx_normalize_inplace_end_ref(basek, &mut x[span << 2..], &mut carry[span << 2..]);
    }
}

#[cfg(all(test, any(target_arch = "x86_64", target_arch = "x86")))]
mod tests {
    use super::*;

    use std::arch::x86_64::_mm256_loadu_si256;

    #[target_feature(enable = "avx2")]
    fn test_get_digit_avx_internal() {
        let basek: usize = 12;
        let x: [i64; 4] = [
            7638646372408325293,
            -61440197422348985,
            6835891051541717957,
            -4835376105455195188,
        ];
        let y0: Vec<i64> = vec![
            get_digit(basek, x[0]),
            get_digit(basek, x[1]),
            get_digit(basek, x[2]),
            get_digit(basek, x[3]),
        ];
        let mut y1: Vec<i64> = vec![0i64; 4];
        unsafe {
            let x_256: __m256i = _mm256_loadu_si256(x.as_ptr() as *const __m256i);
            let (mask, sign, _, _) = normalize_consts_avx(basek);
            let digit: __m256i = get_digit_avx(x_256, mask, sign);
            _mm256_storeu_si256(y1.as_mut_ptr() as *mut __m256i, digit);
        }
        assert_eq!(y0, y1);
    }

    #[test]
    fn test_get_digit_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_get_digit_avx_internal();
        }
    }

    #[target_feature(enable = "avx2")]
    fn test_get_carry_avx_internal() {
        let basek: usize = 12;
        let x: [i64; 4] = [
            7638646372408325293,
            -61440197422348985,
            6835891051541717957,
            -4835376105455195188,
        ];
        let carry: [i64; 4] = [1174467039, -144794816, -1466676977, 513122840];
        let y0: Vec<i64> = vec![
            get_carry(basek, x[0], carry[0]),
            get_carry(basek, x[1], carry[1]),
            get_carry(basek, x[2], carry[2]),
            get_carry(basek, x[3], carry[3]),
        ];
        let mut y1: Vec<i64> = vec![0i64; 4];
        unsafe {
            let x_256: __m256i = _mm256_loadu_si256(x.as_ptr() as *const __m256i);
            let d_256: __m256i = _mm256_loadu_si256(carry.as_ptr() as *const __m256i);
            let (_, _, basek_vec, top_mask) = normalize_consts_avx(basek);
            let digit: __m256i = get_carry_avx(x_256, d_256, basek_vec, top_mask);
            _mm256_storeu_si256(y1.as_mut_ptr() as *mut __m256i, digit);
        }
        assert_eq!(y0, y1);
    }

    #[test]
    fn test_get_carry_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_get_carry_avx_internal();
        }
    }

    #[target_feature(enable = "avx2")]
    fn test_znx_normalize_inplace_beg_avx_internal() {
        let mut y0: [i64; 4] = [
            7638646372408325293,
            -61440197422348985,
            6835891051541717957,
            -4835376105455195188,
        ];
        let mut y1: [i64; 4] = y0;

        let mut c0: [i64; 4] = [
            621182201135793202,
            9000856573317006236,
            5542252755421113668,
            -6036847263131690631,
        ];
        let mut c1: [i64; 4] = c0;

        let basek = 12;

        znx_normalize_inplace_beg_ref(basek, &mut y0, &mut c0);
        znx_normalize_inplace_beg_avx(basek, &mut y1, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_inplace_beg_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_normalize_inplace_beg_avx_internal();
        }
    }

    #[target_feature(enable = "avx2")]
    fn test_znx_normalize_inplace_mid_avx_internal() {
        let mut y0: [i64; 4] = [
            7638646372408325293,
            -61440197422348985,
            6835891051541717957,
            -4835376105455195188,
        ];
        let mut y1: [i64; 4] = y0;

        let mut c0: [i64; 4] = [
            621182201135793202,
            9000856573317006236,
            5542252755421113668,
            -6036847263131690631,
        ];
        let mut c1: [i64; 4] = c0;

        let basek = 12;

        znx_normalize_inplace_mid_ref(basek, &mut y0, &mut c0);
        znx_normalize_inplace_mid_avx(basek, &mut y1, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_inplace_mid_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_normalize_inplace_mid_avx_internal();
        }
    }

    #[target_feature(enable = "avx2")]
    fn test_znx_normalize_inplace_end_avx_internal() {
        let mut y0: [i64; 4] = [
            7638646372408325293,
            -61440197422348985,
            6835891051541717957,
            -4835376105455195188,
        ];
        let mut y1: [i64; 4] = y0;

        let mut c0: [i64; 4] = [
            621182201135793202,
            9000856573317006236,
            5542252755421113668,
            -6036847263131690631,
        ];
        let mut c1: [i64; 4] = c0;

        let basek = 12;

        znx_normalize_inplace_end_ref(basek, &mut y0, &mut c0);
        znx_normalize_inplace_end_avx(basek, &mut y1, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_inplace_end_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_normalize_inplace_end_avx_internal();
        }
    }

    #[target_feature(enable = "avx2")]
    fn test_znx_normalize_beg_avx_internal() {
        let mut y0: [i64; 4] = [
            7638646372408325293,
            -61440197422348985,
            6835891051541717957,
            -4835376105455195188,
        ];
        let mut y1: [i64; 4] = y0;
        let a: [i64; 4] = y0;

        let mut c0: [i64; 4] = [
            621182201135793202,
            9000856573317006236,
            5542252755421113668,
            -6036847263131690631,
        ];
        let mut c1: [i64; 4] = c0;

        let basek = 12;

        znx_normalize_beg_ref(basek, &mut y0, &a, &mut c0);
        znx_normalize_beg_avx(basek, &mut y1, &a, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_beg_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_normalize_beg_avx_internal();
        }
    }

    #[target_feature(enable = "avx2")]
    fn test_znx_normalize_mid_avx_internal() {
        let mut y0: [i64; 4] = [
            7638646372408325293,
            -61440197422348985,
            6835891051541717957,
            -4835376105455195188,
        ];
        let mut y1: [i64; 4] = y0;
        let a: [i64; 4] = y0;

        let mut c0: [i64; 4] = [
            621182201135793202,
            9000856573317006236,
            5542252755421113668,
            -6036847263131690631,
        ];
        let mut c1: [i64; 4] = c0;

        let basek = 12;

        znx_normalize_mid_ref(basek, &mut y0, &a, &mut c0);
        znx_normalize_mid_avx(basek, &mut y1, &a, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_mid_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_normalize_mid_avx_internal();
        }
    }

    #[target_feature(enable = "avx2")]
    fn test_znx_normalize_end_avx_internal() {
        let mut y0: [i64; 4] = [
            7638646372408325293,
            -61440197422348985,
            6835891051541717957,
            -4835376105455195188,
        ];
        let mut y1: [i64; 4] = y0;
        let a: [i64; 4] = y0;

        let mut c0: [i64; 4] = [
            621182201135793202,
            9000856573317006236,
            5542252755421113668,
            -6036847263131690631,
        ];
        let mut c1: [i64; 4] = c0;

        let basek = 12;

        znx_normalize_end_ref(basek, &mut y0, &a, &mut c0);
        znx_normalize_end_avx(basek, &mut y1, &a, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_end_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_normalize_end_avx_internal();
        }
    }
}

#[target_feature(enable = "avx2")]
fn print_m256i_as_i64x4(v: __m256i) {
    let mut a = [0i64; 4];
    unsafe {
        _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, v);
    }
    println!("{a:?}");
}
