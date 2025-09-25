use std::arch::x86_64::__m256i;

/// Vector forms of those constants (broadcast to all lanes)
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
fn normalize_consts_avx(basek: usize) -> (__m256i, __m256i, __m256i, __m256i) {
    use std::arch::x86_64::_mm256_set1_epi64x;

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
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
fn get_digit_avx(x: __m256i, mask_k: __m256i, sign_k: __m256i) -> __m256i {
    use std::arch::x86_64::{_mm256_and_si256, _mm256_sub_epi64, _mm256_xor_si256};
    let low: __m256i = _mm256_and_si256(x, mask_k);
    let t: __m256i = _mm256_xor_si256(low, sign_k);
    _mm256_sub_epi64(t, sign_k)
}

/// AVX2 get_carry using precomputed shift and topmask:
/// carry = (x - digit) >>_arith k
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
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

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
pub fn znx_normalize_first_step_carry_only_avx(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert!(lsh < basek);
    }

    use std::arch::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    unsafe {
        let mut xx: *const __m256i = x.as_ptr() as *const __m256i;
        let mut cc: *mut __m256i = carry.as_ptr() as *mut __m256i;

        let (mask, sign, basek_vec, top_mask) = if lsh == 0 {
            normalize_consts_avx(basek)
        } else {
            normalize_consts_avx(basek - lsh)
        };

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
        use poulpy_hal::reference::znx::znx_normalize_first_step_carry_only_ref;

        znx_normalize_first_step_carry_only_ref(basek, lsh, &x[span << 2..], &mut carry[span << 2..]);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
pub fn znx_normalize_first_step_inplace_avx(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert!(lsh < basek);
    }

    use std::arch::x86_64::{_mm256_loadu_si256, _mm256_set1_epi64x, _mm256_sllv_epi64, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    unsafe {
        let mut xx: *mut __m256i = x.as_mut_ptr() as *mut __m256i;
        let mut cc: *mut __m256i = carry.as_ptr() as *mut __m256i;

        if lsh == 0 {
            let (mask, sign, basek_vec, top_mask) = normalize_consts_avx(basek);

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
        } else {
            let (mask, sign, basek_vec, top_mask) = normalize_consts_avx(basek - lsh);

            let lsh_v: __m256i = _mm256_set1_epi64x(lsh as i64);

            for _ in 0..span {
                let xx_256: __m256i = _mm256_loadu_si256(xx);

                //  (x << (64 - basek)) >> (64 - basek)
                let digit_256: __m256i = get_digit_avx(xx_256, mask, sign);

                // (x - digit) >> basek
                let carry_256: __m256i = get_carry_avx(xx_256, digit_256, basek_vec, top_mask);

                _mm256_storeu_si256(xx, _mm256_sllv_epi64(digit_256, lsh_v));
                _mm256_storeu_si256(cc, carry_256);

                xx = xx.add(1);
                cc = cc.add(1);
            }
        }
    }

    // tail
    if !x.len().is_multiple_of(4) {
        use poulpy_hal::reference::znx::znx_normalize_first_step_inplace_ref;

        znx_normalize_first_step_inplace_ref(basek, lsh, &mut x[span << 2..], &mut carry[span << 2..]);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
pub fn znx_normalize_first_step_avx(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert_eq!(a.len(), carry.len());
        assert!(lsh < basek);
    }

    use std::arch::x86_64::{_mm256_loadu_si256, _mm256_sllv_epi64, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    unsafe {
        let mut xx: *mut __m256i = x.as_mut_ptr() as *mut __m256i;
        let mut aa: *const __m256i = a.as_ptr() as *const __m256i;
        let mut cc: *mut __m256i = carry.as_ptr() as *mut __m256i;

        if lsh == 0 {
            let (mask, sign, basek_vec, top_mask) = normalize_consts_avx(basek);

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
        } else {
            use std::arch::x86_64::_mm256_set1_epi64x;

            let (mask, sign, basek_vec, top_mask) = normalize_consts_avx(basek - lsh);

            let lsh_v: __m256i = _mm256_set1_epi64x(lsh as i64);

            for _ in 0..span {
                let aa_256: __m256i = _mm256_loadu_si256(aa);

                //  (x << (64 - basek)) >> (64 - basek)
                let digit_256: __m256i = get_digit_avx(aa_256, mask, sign);

                // (x - digit) >> basek
                let carry_256: __m256i = get_carry_avx(aa_256, digit_256, basek_vec, top_mask);

                _mm256_storeu_si256(xx, _mm256_sllv_epi64(digit_256, lsh_v));
                _mm256_storeu_si256(cc, carry_256);

                xx = xx.add(1);
                aa = aa.add(1);
                cc = cc.add(1);
            }
        }
    }

    // tail
    if !x.len().is_multiple_of(4) {
        use poulpy_hal::reference::znx::znx_normalize_first_step_ref;

        znx_normalize_first_step_ref(
            basek,
            lsh,
            &mut x[span << 2..],
            &a[span << 2..],
            &mut carry[span << 2..],
        );
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
pub fn znx_normalize_middle_step_inplace_avx(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert!(lsh < basek);
    }

    use std::arch::x86_64::{_mm256_add_epi64, _mm256_loadu_si256, _mm256_sllv_epi64, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    let (mask, sign, basek_vec, top_mask) = normalize_consts_avx(basek);

    unsafe {
        let mut xx: *mut __m256i = x.as_mut_ptr() as *mut __m256i;
        let mut cc: *mut __m256i = carry.as_mut_ptr() as *mut __m256i;

        if lsh == 0 {
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
        } else {
            use std::arch::x86_64::_mm256_set1_epi64x;

            let (mask_lsh, sign_lsh, basek_vec_lsh, top_mask_lsh) = normalize_consts_avx(basek - lsh);

            let lsh_v: __m256i = _mm256_set1_epi64x(lsh as i64);

            for _ in 0..span {
                let xx_256: __m256i = _mm256_loadu_si256(xx);
                let cc_256: __m256i = _mm256_loadu_si256(cc);

                let d0: __m256i = get_digit_avx(xx_256, mask_lsh, sign_lsh);
                let c0: __m256i = get_carry_avx(xx_256, d0, basek_vec_lsh, top_mask_lsh);

                let d0_lsh: __m256i = _mm256_sllv_epi64(d0, lsh_v);

                let s: __m256i = _mm256_add_epi64(d0_lsh, cc_256);
                let x1: __m256i = get_digit_avx(s, mask, sign);
                let c1: __m256i = get_carry_avx(s, x1, basek_vec, top_mask);
                let cout: __m256i = _mm256_add_epi64(c0, c1);

                _mm256_storeu_si256(xx, x1);
                _mm256_storeu_si256(cc, cout);

                xx = xx.add(1);
                cc = cc.add(1);
            }
        }
    }

    if !x.len().is_multiple_of(4) {
        use poulpy_hal::reference::znx::znx_normalize_middle_step_inplace_ref;

        znx_normalize_middle_step_inplace_ref(basek, lsh, &mut x[span << 2..], &mut carry[span << 2..]);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
pub fn znx_normalize_middle_step_carry_only_avx(basek: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert!(lsh < basek);
    }

    use std::arch::x86_64::{_mm256_add_epi64, _mm256_loadu_si256, _mm256_sllv_epi64, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    let (mask, sign, basek_vec, top_mask) = normalize_consts_avx(basek);

    unsafe {
        let mut xx: *const __m256i = x.as_ptr() as *const __m256i;
        let mut cc: *mut __m256i = carry.as_mut_ptr() as *mut __m256i;

        if lsh == 0 {
            for _ in 0..span {
                let xx_256: __m256i = _mm256_loadu_si256(xx);
                let cc_256: __m256i = _mm256_loadu_si256(cc);

                let d0: __m256i = get_digit_avx(xx_256, mask, sign);
                let c0: __m256i = get_carry_avx(xx_256, d0, basek_vec, top_mask);

                let s: __m256i = _mm256_add_epi64(d0, cc_256);
                let x1: __m256i = get_digit_avx(s, mask, sign);
                let c1: __m256i = get_carry_avx(s, x1, basek_vec, top_mask);
                let cout: __m256i = _mm256_add_epi64(c0, c1);

                _mm256_storeu_si256(cc, cout);

                xx = xx.add(1);
                cc = cc.add(1);
            }
        } else {
            use std::arch::x86_64::_mm256_set1_epi64x;

            let (mask_lsh, sign_lsh, basek_vec_lsh, top_mask_lsh) = normalize_consts_avx(basek - lsh);

            let lsh_v: __m256i = _mm256_set1_epi64x(lsh as i64);

            for _ in 0..span {
                let xx_256: __m256i = _mm256_loadu_si256(xx);
                let cc_256: __m256i = _mm256_loadu_si256(cc);

                let d0: __m256i = get_digit_avx(xx_256, mask_lsh, sign_lsh);
                let c0: __m256i = get_carry_avx(xx_256, d0, basek_vec_lsh, top_mask_lsh);

                let d0_lsh: __m256i = _mm256_sllv_epi64(d0, lsh_v);

                let s: __m256i = _mm256_add_epi64(d0_lsh, cc_256);
                let x1: __m256i = get_digit_avx(s, mask, sign);
                let c1: __m256i = get_carry_avx(s, x1, basek_vec, top_mask);
                let cout: __m256i = _mm256_add_epi64(c0, c1);

                _mm256_storeu_si256(cc, cout);

                xx = xx.add(1);
                cc = cc.add(1);
            }
        }
    }

    if !x.len().is_multiple_of(4) {
        use poulpy_hal::reference::znx::znx_normalize_middle_step_carry_only_ref;

        znx_normalize_middle_step_carry_only_ref(basek, lsh, &x[span << 2..], &mut carry[span << 2..]);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
pub fn znx_normalize_middle_step_avx(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert_eq!(a.len(), carry.len());
        assert!(lsh < basek);
    }

    use std::arch::x86_64::{_mm256_add_epi64, _mm256_loadu_si256, _mm256_sllv_epi64, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    let (mask, sign, basek_vec, top_mask) = normalize_consts_avx(basek);

    unsafe {
        let mut xx: *mut __m256i = x.as_mut_ptr() as *mut __m256i;
        let mut aa: *const __m256i = a.as_ptr() as *const __m256i;
        let mut cc: *mut __m256i = carry.as_ptr() as *mut __m256i;

        if lsh == 0 {
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
        } else {
            use std::arch::x86_64::_mm256_set1_epi64x;

            let (mask_lsh, sign_lsh, basek_vec_lsh, top_mask_lsh) = normalize_consts_avx(basek - lsh);

            let lsh_v: __m256i = _mm256_set1_epi64x(lsh as i64);

            for _ in 0..span {
                let aa_256: __m256i = _mm256_loadu_si256(aa);
                let cc_256: __m256i = _mm256_loadu_si256(cc);

                let d0: __m256i = get_digit_avx(aa_256, mask_lsh, sign_lsh);
                let c0: __m256i = get_carry_avx(aa_256, d0, basek_vec_lsh, top_mask_lsh);

                let d0_lsh: __m256i = _mm256_sllv_epi64(d0, lsh_v);

                let s: __m256i = _mm256_add_epi64(d0_lsh, cc_256);
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
    }

    if !x.len().is_multiple_of(4) {
        use poulpy_hal::reference::znx::znx_normalize_middle_step_ref;

        znx_normalize_middle_step_ref(
            basek,
            lsh,
            &mut x[span << 2..],
            &a[span << 2..],
            &mut carry[span << 2..],
        );
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
pub fn znx_normalize_final_step_inplace_avx(basek: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert!(lsh < basek);
    }

    use std::arch::x86_64::{_mm256_add_epi64, _mm256_loadu_si256, _mm256_sllv_epi64, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    let (mask, sign, _, _) = normalize_consts_avx(basek);

    unsafe {
        let mut xx: *mut __m256i = x.as_mut_ptr() as *mut __m256i;
        let mut cc: *mut __m256i = carry.as_ptr() as *mut __m256i;

        if lsh == 0 {
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
        } else {
            use std::arch::x86_64::_mm256_set1_epi64x;

            let (mask_lsh, sign_lsh, _, _) = normalize_consts_avx(basek - lsh);

            let lsh_v: __m256i = _mm256_set1_epi64x(lsh as i64);

            for _ in 0..span {
                let xv: __m256i = _mm256_loadu_si256(xx);
                let cv: __m256i = _mm256_loadu_si256(cc);

                let d0: __m256i = get_digit_avx(xv, mask_lsh, sign_lsh);

                let d0_lsh: __m256i = _mm256_sllv_epi64(d0, lsh_v);

                let s: __m256i = _mm256_add_epi64(d0_lsh, cv);
                let x1: __m256i = get_digit_avx(s, mask, sign);

                _mm256_storeu_si256(xx, x1);

                xx = xx.add(1);
                cc = cc.add(1);
            }
        }
    }

    if !x.len().is_multiple_of(4) {
        use poulpy_hal::reference::znx::znx_normalize_final_step_inplace_ref;

        znx_normalize_final_step_inplace_ref(basek, lsh, &mut x[span << 2..], &mut carry[span << 2..]);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
pub fn znx_normalize_final_step_avx(basek: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), carry.len());
        assert_eq!(a.len(), carry.len());
        assert!(lsh < basek);
    }

    use std::arch::x86_64::{_mm256_add_epi64, _mm256_loadu_si256, _mm256_sllv_epi64, _mm256_storeu_si256};

    let n: usize = x.len();

    let span: usize = n >> 2;

    let (mask, sign, _, _) = normalize_consts_avx(basek);

    unsafe {
        let mut xx: *mut __m256i = x.as_mut_ptr() as *mut __m256i;
        let mut aa: *mut __m256i = a.as_ptr() as *mut __m256i;
        let mut cc: *mut __m256i = carry.as_ptr() as *mut __m256i;

        if lsh == 0 {
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
        } else {
            use std::arch::x86_64::_mm256_set1_epi64x;

            let (mask_lsh, sign_lsh, _, _) = normalize_consts_avx(basek - lsh);

            let lsh_v: __m256i = _mm256_set1_epi64x(lsh as i64);

            for _ in 0..span {
                let av: __m256i = _mm256_loadu_si256(aa);
                let cv: __m256i = _mm256_loadu_si256(cc);

                let d0: __m256i = get_digit_avx(av, mask_lsh, sign_lsh);
                let d0_lsh: __m256i = _mm256_sllv_epi64(d0, lsh_v);

                let s: __m256i = _mm256_add_epi64(d0_lsh, cv);
                let x1: __m256i = get_digit_avx(s, mask, sign);

                _mm256_storeu_si256(xx, x1);

                xx = xx.add(1);
                aa = aa.add(1);
                cc = cc.add(1);
            }
        }
    }

    if !x.len().is_multiple_of(4) {
        use poulpy_hal::reference::znx::znx_normalize_final_step_ref;

        znx_normalize_final_step_ref(
            basek,
            lsh,
            &mut x[span << 2..],
            &a[span << 2..],
            &mut carry[span << 2..],
        );
    }
}

mod tests {
    use poulpy_hal::reference::znx::{
        get_carry, get_digit, znx_normalize_final_step_inplace_ref, znx_normalize_final_step_ref,
        znx_normalize_first_step_inplace_ref, znx_normalize_first_step_ref, znx_normalize_middle_step_inplace_ref,
        znx_normalize_middle_step_ref,
    };

    use super::*;

    use std::arch::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256};

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
    fn test_znx_normalize_first_step_inplace_avx_internal() {
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

        znx_normalize_first_step_inplace_ref(basek, 0, &mut y0, &mut c0);
        znx_normalize_first_step_inplace_avx(basek, 0, &mut y1, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);

        znx_normalize_first_step_inplace_ref(basek, basek - 1, &mut y0, &mut c0);
        znx_normalize_first_step_inplace_avx(basek, basek - 1, &mut y1, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_first_step_inplace_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_normalize_first_step_inplace_avx_internal();
        }
    }

    #[target_feature(enable = "avx2")]
    fn test_znx_normalize_middle_step_inplace_avx_internal() {
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

        znx_normalize_middle_step_inplace_ref(basek, 0, &mut y0, &mut c0);
        znx_normalize_middle_step_inplace_avx(basek, 0, &mut y1, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);

        znx_normalize_middle_step_inplace_ref(basek, basek - 1, &mut y0, &mut c0);
        znx_normalize_middle_step_inplace_avx(basek, basek - 1, &mut y1, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_middle_step_inplace_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_normalize_middle_step_inplace_avx_internal();
        }
    }

    #[target_feature(enable = "avx2")]
    fn test_znx_normalize_final_step_inplace_avx_internal() {
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

        znx_normalize_final_step_inplace_ref(basek, 0, &mut y0, &mut c0);
        znx_normalize_final_step_inplace_avx(basek, 0, &mut y1, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);

        znx_normalize_final_step_inplace_ref(basek, basek - 1, &mut y0, &mut c0);
        znx_normalize_final_step_inplace_avx(basek, basek - 1, &mut y1, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_final_step_inplace_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_normalize_final_step_inplace_avx_internal();
        }
    }

    #[target_feature(enable = "avx2")]
    fn test_znx_normalize_first_step_avx_internal() {
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

        znx_normalize_first_step_ref(basek, 0, &mut y0, &a, &mut c0);
        znx_normalize_first_step_avx(basek, 0, &mut y1, &a, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);

        znx_normalize_first_step_ref(basek, basek - 1, &mut y0, &a, &mut c0);
        znx_normalize_first_step_avx(basek, basek - 1, &mut y1, &a, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_first_step_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_normalize_first_step_avx_internal();
        }
    }

    #[target_feature(enable = "avx2")]
    fn test_znx_normalize_middle_step_avx_internal() {
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

        znx_normalize_middle_step_ref(basek, 0, &mut y0, &a, &mut c0);
        znx_normalize_middle_step_avx(basek, 0, &mut y1, &a, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);

        znx_normalize_middle_step_ref(basek, basek - 1, &mut y0, &a, &mut c0);
        znx_normalize_middle_step_avx(basek, basek - 1, &mut y1, &a, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_middle_step_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_normalize_middle_step_avx_internal();
        }
    }

    #[target_feature(enable = "avx2")]
    fn test_znx_normalize_final_step_avx_internal() {
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

        znx_normalize_final_step_ref(basek, 0, &mut y0, &a, &mut c0);
        znx_normalize_final_step_avx(basek, 0, &mut y1, &a, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);

        znx_normalize_final_step_ref(basek, basek - 1, &mut y0, &a, &mut c0);
        znx_normalize_final_step_avx(basek, basek - 1, &mut y1, &a, &mut c1);

        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_final_step_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_normalize_final_step_avx_internal();
        }
    }
}
