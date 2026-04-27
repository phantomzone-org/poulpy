// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been directly ported from the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The porting process from C to Rust was done with minimal changes
// in order to preserve the semantics and performance characteristics
// of the original implementation.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
#[target_feature(enable = "avx512f")]
pub fn reim4_extract_1blk_from_reim_contiguous_ifma(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_loadu_pd, _mm256_storeu_pd};

    unsafe {
        let mut src_ptr: *const __m256d = src.as_ptr().add(blk << 2) as *const __m256d; // src + 4*blk
        let mut dst_ptr: *mut __m256d = dst.as_mut_ptr() as *mut __m256d;

        let step: usize = m >> 2;

        // Each iteration copies 4 doubles; advance src by m doubles each row
        for _ in 0..2 * rows {
            let v: __m256d = _mm256_loadu_pd(src_ptr as *const f64);
            _mm256_storeu_pd(dst_ptr as *mut f64, v);
            dst_ptr = dst_ptr.add(1);
            src_ptr = src_ptr.add(step);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
#[target_feature(enable = "avx512f")]
pub fn reim4_save_1blk_to_reim_contiguous_ifma(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_loadu_pd, _mm256_storeu_pd};

    unsafe {
        let mut src_ptr: *const __m256d = src.as_ptr() as *const __m256d;
        let mut dst_ptr: *mut __m256d = dst.as_mut_ptr().add(blk << 2) as *mut __m256d; // dst + 4*blk

        let step: usize = m >> 2;

        // Each iteration copies 4 doubles; advance dst by m doubles each row
        for _ in 0..2 * rows {
            let v: __m256d = _mm256_loadu_pd(src_ptr as *const f64);
            _mm256_storeu_pd(dst_ptr as *mut f64, v);
            dst_ptr = dst_ptr.add(step);
            src_ptr = src_ptr.add(1);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
#[target_feature(enable = "avx512f")]
pub fn reim4_save_1blk_to_reim_ifma<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};
    unsafe {
        let off: usize = blk * 4;
        let src_ptr: *const f64 = src.as_ptr();

        let s0: __m256d = _mm256_loadu_pd(src_ptr);
        let s1: __m256d = _mm256_loadu_pd(src_ptr.add(4));

        let d0_ptr: *mut f64 = dst.as_mut_ptr().add(off);
        let d1_ptr: *mut f64 = d0_ptr.add(m);

        if OVERWRITE {
            _mm256_storeu_pd(d0_ptr, s0);
            _mm256_storeu_pd(d1_ptr, s1);
        } else {
            let d0: __m256d = _mm256_loadu_pd(d0_ptr);
            let d1: __m256d = _mm256_loadu_pd(d1_ptr);
            _mm256_storeu_pd(d0_ptr, _mm256_add_pd(d0, s0));
            _mm256_storeu_pd(d1_ptr, _mm256_add_pd(d1, s1));
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
#[target_feature(enable = "avx512f")]
pub fn reim4_save_2blk_to_reim_ifma<const OVERWRITE: bool>(
    m: usize,        //
    blk: usize,      // block index
    dst: &mut [f64], //
    src: &[f64],     // 16 doubles [re1(4), im1(4), re2(4), im2(4)]
) {
    use core::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};
    unsafe {
        let off: usize = blk * 4;
        let src_ptr: *const f64 = src.as_ptr();

        let d0_ptr: *mut f64 = dst.as_mut_ptr().add(off);
        let d1_ptr: *mut f64 = d0_ptr.add(m);
        let d2_ptr: *mut f64 = d1_ptr.add(m);
        let d3_ptr: *mut f64 = d2_ptr.add(m);

        let s0: __m256d = _mm256_loadu_pd(src_ptr);
        let s1: __m256d = _mm256_loadu_pd(src_ptr.add(4));
        let s2: __m256d = _mm256_loadu_pd(src_ptr.add(8));
        let s3: __m256d = _mm256_loadu_pd(src_ptr.add(12));

        if OVERWRITE {
            _mm256_storeu_pd(d0_ptr, s0);
            _mm256_storeu_pd(d1_ptr, s1);
            _mm256_storeu_pd(d2_ptr, s2);
            _mm256_storeu_pd(d3_ptr, s3);
        } else {
            let d0: __m256d = _mm256_loadu_pd(d0_ptr);
            let d1: __m256d = _mm256_loadu_pd(d1_ptr);
            let d2: __m256d = _mm256_loadu_pd(d2_ptr);
            let d3: __m256d = _mm256_loadu_pd(d3_ptr);
            _mm256_storeu_pd(d0_ptr, _mm256_add_pd(d0, s0));
            _mm256_storeu_pd(d1_ptr, _mm256_add_pd(d1, s1));
            _mm256_storeu_pd(d2_ptr, _mm256_add_pd(d2, s2));
            _mm256_storeu_pd(d3_ptr, _mm256_add_pd(d3, s3));
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
#[target_feature(enable = "avx512f")]
pub fn reim4_vec_mat1col_product_ifma(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
    use core::arch::x86_64::{
        __m256d, __m512d, _mm256_add_pd, _mm256_storeu_pd, _mm256_sub_pd, _mm512_castpd512_pd256, _mm512_extractf64x4_pd,
        _mm512_fmadd_pd, _mm512_loadu_pd, _mm512_setzero_pd, _mm512_shuffle_f64x2,
    };

    #[cfg(debug_assertions)]
    {
        assert!(dst.len() >= 8, "dst must have at least 8 doubles");
        assert!(u.len() >= nrows * 8, "u must be at least nrows * 8 doubles");
        assert!(v.len() >= nrows * 8, "v must be at least nrows * 8 doubles");
    }

    unsafe {
        // Packed accumulators:
        //   acc_a = [re1 | re2]  (low 256 bits = re1, high 256 bits = re2)
        //   acc_b = [im1 | im2]
        let mut acc_a: __m512d = _mm512_setzero_pd();
        let mut acc_b: __m512d = _mm512_setzero_pd();

        let mut u_ptr: *const f64 = u.as_ptr();
        let mut v_ptr: *const f64 = v.as_ptr();

        for _ in 0..nrows {
            let u_full: __m512d = _mm512_loadu_pd(u_ptr); // [ur | ui]
            let v_full: __m512d = _mm512_loadu_pd(v_ptr); // [vr | vi]
            // Swap 256-bit halves: [vi | vr]
            let v_swap: __m512d = _mm512_shuffle_f64x2::<0b01_00_11_10>(v_full, v_full);

            // acc_a += u_full * v_full  =>  re1 += ur*vr;  re2 += ui*vi
            acc_a = _mm512_fmadd_pd(u_full, v_full, acc_a);
            // acc_b += u_full * v_swap  =>  im1 += ur*vi;  im2 += ui*vr
            acc_b = _mm512_fmadd_pd(u_full, v_swap, acc_b);

            u_ptr = u_ptr.add(8);
            v_ptr = v_ptr.add(8);
        }

        let re1: __m256d = _mm512_castpd512_pd256(acc_a);
        let re2: __m256d = _mm512_extractf64x4_pd::<1>(acc_a);
        let im1: __m256d = _mm512_castpd512_pd256(acc_b);
        let im2: __m256d = _mm512_extractf64x4_pd::<1>(acc_b);

        // re1 - re2
        _mm256_storeu_pd(dst.as_mut_ptr(), _mm256_sub_pd(re1, re2));
        // im1 + im2
        _mm256_storeu_pd(dst.as_mut_ptr().add(4), _mm256_add_pd(im1, im2));
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
#[target_feature(enable = "avx512f")]
pub fn reim4_vec_mat2cols_product_ifma(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
    use core::arch::x86_64::{
        __m256d, __m512d, _mm256_storeu_pd, _mm512_castpd512_pd256, _mm512_extractf64x4_pd, _mm512_fmadd_pd, _mm512_fnmadd_pd,
        _mm512_loadu_pd, _mm512_setzero_pd, _mm512_shuffle_f64x2,
    };

    #[cfg(debug_assertions)]
    {
        assert!(dst.len() >= 8, "dst must be at least 8 doubles but is {}", dst.len());
        assert!(
            u.len() >= nrows * 8,
            "u must be at least nrows={} * 8 doubles but is {}",
            nrows,
            u.len()
        );
        assert!(
            v.len() >= nrows * 16,
            "v must be at least nrows={} * 16 doubles but is {}",
            nrows,
            v.len()
        );
    }

    unsafe {
        // Packed accumulators:
        //   acc_re = [re1 | re2]  (col0_re | col1_re)
        //   acc_im = [im1 | im2]  (col0_im | col1_im)
        let mut acc_re: __m512d = _mm512_setzero_pd();
        let mut acc_im: __m512d = _mm512_setzero_pd();

        let mut u_ptr: *const f64 = u.as_ptr();
        let mut v_ptr: *const f64 = v.as_ptr();

        for _ in 0..nrows {
            // u_full = [ur(4) | ui(4)]
            let u_full: __m512d = _mm512_loadu_pd(u_ptr);
            // Broadcast each half across both halves: [ur | ur] and [ui | ui]
            let ur_dup: __m512d = _mm512_shuffle_f64x2::<0b01_00_01_00>(u_full, u_full);
            let ui_dup: __m512d = _mm512_shuffle_f64x2::<0b11_10_11_10>(u_full, u_full);

            // va = [ar(4) | ai(4)], vb = [br(4) | bi(4)]
            let va: __m512d = _mm512_loadu_pd(v_ptr);
            let vb: __m512d = _mm512_loadu_pd(v_ptr.add(8));
            // v_re = [ar | br], v_im = [ai | bi]
            let v_re: __m512d = _mm512_shuffle_f64x2::<0b01_00_01_00>(va, vb);
            let v_im: __m512d = _mm512_shuffle_f64x2::<0b11_10_11_10>(va, vb);

            // re += ur*v_re - ui*v_im  (low: re1 = ur*ar - ui*ai; high: re2 = ur*br - ui*bi)
            acc_re = _mm512_fmadd_pd(ur_dup, v_re, acc_re);
            acc_re = _mm512_fnmadd_pd(ui_dup, v_im, acc_re);
            // im += ur*v_im + ui*v_re  (low: im1 = ur*ai + ui*ar; high: im2 = ur*bi + ui*br)
            acc_im = _mm512_fmadd_pd(ur_dup, v_im, acc_im);
            acc_im = _mm512_fmadd_pd(ui_dup, v_re, acc_im);

            u_ptr = u_ptr.add(8);
            v_ptr = v_ptr.add(16);
        }

        let re1: __m256d = _mm512_castpd512_pd256(acc_re);
        let re2: __m256d = _mm512_extractf64x4_pd::<1>(acc_re);
        let im1: __m256d = _mm512_castpd512_pd256(acc_im);
        let im2: __m256d = _mm512_extractf64x4_pd::<1>(acc_im);

        _mm256_storeu_pd(dst.as_mut_ptr(), re1);
        _mm256_storeu_pd(dst.as_mut_ptr().add(4), im1);
        _mm256_storeu_pd(dst.as_mut_ptr().add(8), re2);
        _mm256_storeu_pd(dst.as_mut_ptr().add(12), im2);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
#[target_feature(enable = "avx512f")]
pub fn reim4_vec_mat2cols_2ndcol_product_ifma(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
    use core::arch::x86_64::{
        __m256d, __m512d, _mm256_add_pd, _mm256_storeu_pd, _mm256_sub_pd, _mm512_castpd512_pd256, _mm512_extractf64x4_pd,
        _mm512_fmadd_pd, _mm512_loadu_pd, _mm512_setzero_pd, _mm512_shuffle_f64x2,
    };

    #[cfg(debug_assertions)]
    {
        assert_eq!(dst.len(), 16, "dst must have 16 doubles");
        assert!(u.len() >= nrows * 8, "u must be at least nrows * 8 doubles");
        assert!(v.len() >= nrows * 16, "v must be at least nrows * 16 doubles");
    }

    unsafe {
        // Packed accumulators: acc_a = [Σ ur*ar | Σ ui*ai], acc_b = [Σ ur*ai | Σ ui*ar]
        let mut acc_a: __m512d = _mm512_setzero_pd();
        let mut acc_b: __m512d = _mm512_setzero_pd();

        let mut u_ptr: *const f64 = u.as_ptr();
        let mut v_ptr: *const f64 = v.as_ptr().add(8); // Offset to 2nd column

        for _ in 0..nrows {
            let u_full: __m512d = _mm512_loadu_pd(u_ptr); // [ur(4) | ui(4)]
            let v_full: __m512d = _mm512_loadu_pd(v_ptr); // [ar(4) | ai(4)]
            let v_swap: __m512d = _mm512_shuffle_f64x2::<0b01_00_11_10>(v_full, v_full); // [ai(4) | ar(4)]

            acc_a = _mm512_fmadd_pd(u_full, v_full, acc_a); // [ur*ar | ui*ai]
            acc_b = _mm512_fmadd_pd(u_full, v_swap, acc_b); // [ur*ai | ui*ar]

            u_ptr = u_ptr.add(8);
            v_ptr = v_ptr.add(16);
        }

        // re = Σ ur*ar - Σ ui*ai, im = Σ ur*ai + Σ ui*ar
        let lo_a: __m256d = _mm512_castpd512_pd256(acc_a);
        let hi_a: __m256d = _mm512_extractf64x4_pd::<1>(acc_a);
        let lo_b: __m256d = _mm512_castpd512_pd256(acc_b);
        let hi_b: __m256d = _mm512_extractf64x4_pd::<1>(acc_b);

        _mm256_storeu_pd(dst.as_mut_ptr(), _mm256_sub_pd(lo_a, hi_a));
        _mm256_storeu_pd(dst.as_mut_ptr().add(4), _mm256_add_pd(lo_b, hi_b));
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g. `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub unsafe fn reim4_convolution_1coeff_ifma(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
    use core::arch::x86_64::{
        __m256d, __m512d, _mm256_add_pd, _mm256_storeu_pd, _mm256_sub_pd, _mm512_castpd512_pd256, _mm512_extractf64x4_pd,
        _mm512_fmadd_pd, _mm512_loadu_pd, _mm512_setzero_pd, _mm512_shuffle_f64x2, _mm512_storeu_pd,
    };

    unsafe {
        if k >= a_size + b_size {
            _mm512_storeu_pd(dst.as_mut_ptr(), _mm512_setzero_pd());
            return;
        }

        let j_min: usize = k.saturating_sub(a_size - 1);
        let j_max: usize = (k + 1).min(b_size);

        // Packed accumulators: acc_a = [Σ ar*br | Σ ai*bi], acc_b = [Σ ar*bi | Σ ai*br]
        let mut acc_a: __m512d = _mm512_setzero_pd();
        let mut acc_b: __m512d = _mm512_setzero_pd();

        let mut a_ptr: *const f64 = a.as_ptr().add(8 * (k - j_min));
        let mut b_ptr: *const f64 = b.as_ptr().add(8 * j_min);

        for _ in 0..j_max - j_min {
            let a_full: __m512d = _mm512_loadu_pd(a_ptr); // [ar(4) | ai(4)]
            let b_full: __m512d = _mm512_loadu_pd(b_ptr); // [br(4) | bi(4)]
            let b_swap: __m512d = _mm512_shuffle_f64x2::<0b01_00_11_10>(b_full, b_full); // [bi(4) | br(4)]

            acc_a = _mm512_fmadd_pd(a_full, b_full, acc_a); // [ar*br | ai*bi]
            acc_b = _mm512_fmadd_pd(a_full, b_swap, acc_b); // [ar*bi | ai*br]

            a_ptr = a_ptr.sub(8);
            b_ptr = b_ptr.add(8);
        }

        // re = Σ ar*br - Σ ai*bi, im = Σ ar*bi + Σ ai*br
        let lo_a: __m256d = _mm512_castpd512_pd256(acc_a);
        let hi_a: __m256d = _mm512_extractf64x4_pd::<1>(acc_a);
        let lo_b: __m256d = _mm512_castpd512_pd256(acc_b);
        let hi_b: __m256d = _mm512_extractf64x4_pd::<1>(acc_b);

        _mm256_storeu_pd(dst.as_mut_ptr(), _mm256_sub_pd(lo_a, hi_a));
        _mm256_storeu_pd(dst.as_mut_ptr().add(4), _mm256_add_pd(lo_b, hi_b));
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g. `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub unsafe fn reim4_convolution_2coeffs_ifma(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
    use core::arch::x86_64::{
        __m256d, __m512d, _mm256_add_pd, _mm256_storeu_pd, _mm256_sub_pd, _mm512_castpd512_pd256, _mm512_extractf64x4_pd,
        _mm512_fmadd_pd, _mm512_loadu_pd, _mm512_setzero_pd, _mm512_shuffle_f64x2, _mm512_storeu_pd,
    };

    debug_assert!(a.len() >= 8 * a_size);
    debug_assert!(b.len() >= 8 * b_size);

    let k0: usize = k;
    let k1: usize = k + 1;
    let bound: usize = a_size + b_size;

    if k0 >= bound {
        unsafe {
            let zero: __m512d = _mm512_setzero_pd();
            _mm512_storeu_pd(dst.as_mut_ptr(), zero);
            _mm512_storeu_pd(dst.as_mut_ptr().add(8), zero);
        }
        return;
    }

    unsafe {
        // Packed accumulators per coefficient: acc_a = [Σ ar*br | Σ ai*bi], acc_b = [Σ ar*bi | Σ ai*br]
        let mut acc_a_k0: __m512d = _mm512_setzero_pd();
        let mut acc_b_k0: __m512d = _mm512_setzero_pd();
        let mut acc_a_k1: __m512d = _mm512_setzero_pd();
        let mut acc_b_k1: __m512d = _mm512_setzero_pd();

        let j0_min: usize = (k0 + 1).saturating_sub(a_size);
        let j0_max: usize = (k0 + 1).min(b_size);

        if k1 >= bound {
            let mut a_k0_ptr: *const f64 = a.as_ptr().add(8 * (k0 - j0_min));
            let mut b_ptr: *const f64 = b.as_ptr().add(8 * j0_min);

            for _ in 0..j0_max - j0_min {
                let a0_full: __m512d = _mm512_loadu_pd(a_k0_ptr);
                let b_full: __m512d = _mm512_loadu_pd(b_ptr);
                let b_swap: __m512d = _mm512_shuffle_f64x2::<0b01_00_11_10>(b_full, b_full);

                acc_a_k0 = _mm512_fmadd_pd(a0_full, b_full, acc_a_k0);
                acc_b_k0 = _mm512_fmadd_pd(a0_full, b_swap, acc_b_k0);

                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(8);
            }
        } else {
            let j1_min: usize = (k1 + 1).saturating_sub(a_size);
            let j1_max: usize = (k1 + 1).min(b_size);

            let mut a_k0_ptr: *const f64 = a.as_ptr().add(8 * (k0 - j0_min));
            let mut a_k1_ptr: *const f64 = a.as_ptr().add(8 * (k1 - j1_min));
            let mut b_ptr: *const f64 = b.as_ptr().add(8 * j0_min);

            // Region 1: contributions to k0 only, j ∈ [j0_min, j1_min)
            for _ in 0..j1_min - j0_min {
                let a0_full: __m512d = _mm512_loadu_pd(a_k0_ptr);
                let b_full: __m512d = _mm512_loadu_pd(b_ptr);
                let b_swap: __m512d = _mm512_shuffle_f64x2::<0b01_00_11_10>(b_full, b_full);

                acc_a_k0 = _mm512_fmadd_pd(a0_full, b_full, acc_a_k0);
                acc_b_k0 = _mm512_fmadd_pd(a0_full, b_swap, acc_b_k0);

                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(8);
            }

            // Region 2: overlap, contributions to both k0 and k1, j ∈ [j1_min, j0_max)
            for _ in 0..j0_max - j1_min {
                let a0_full: __m512d = _mm512_loadu_pd(a_k0_ptr);
                let a1_full: __m512d = _mm512_loadu_pd(a_k1_ptr);
                let b_full: __m512d = _mm512_loadu_pd(b_ptr);
                let b_swap: __m512d = _mm512_shuffle_f64x2::<0b01_00_11_10>(b_full, b_full);

                acc_a_k0 = _mm512_fmadd_pd(a0_full, b_full, acc_a_k0);
                acc_b_k0 = _mm512_fmadd_pd(a0_full, b_swap, acc_b_k0);
                acc_a_k1 = _mm512_fmadd_pd(a1_full, b_full, acc_a_k1);
                acc_b_k1 = _mm512_fmadd_pd(a1_full, b_swap, acc_b_k1);

                a_k0_ptr = a_k0_ptr.sub(8);
                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(8);
            }

            // Region 3: contributions to k1 only, j ∈ [j0_max, j1_max)
            for _ in 0..j1_max - j0_max {
                let a1_full: __m512d = _mm512_loadu_pd(a_k1_ptr);
                let b_full: __m512d = _mm512_loadu_pd(b_ptr);
                let b_swap: __m512d = _mm512_shuffle_f64x2::<0b01_00_11_10>(b_full, b_full);

                acc_a_k1 = _mm512_fmadd_pd(a1_full, b_full, acc_a_k1);
                acc_b_k1 = _mm512_fmadd_pd(a1_full, b_swap, acc_b_k1);

                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(8);
            }
        }

        // re = Σ ar*br - Σ ai*bi, im = Σ ar*bi + Σ ai*br
        let dst_ptr = dst.as_mut_ptr();

        let lo_a0: __m256d = _mm512_castpd512_pd256(acc_a_k0);
        let hi_a0: __m256d = _mm512_extractf64x4_pd::<1>(acc_a_k0);
        let lo_b0: __m256d = _mm512_castpd512_pd256(acc_b_k0);
        let hi_b0: __m256d = _mm512_extractf64x4_pd::<1>(acc_b_k0);
        _mm256_storeu_pd(dst_ptr, _mm256_sub_pd(lo_a0, hi_a0));
        _mm256_storeu_pd(dst_ptr.add(4), _mm256_add_pd(lo_b0, hi_b0));

        let lo_a1: __m256d = _mm512_castpd512_pd256(acc_a_k1);
        let hi_a1: __m256d = _mm512_extractf64x4_pd::<1>(acc_a_k1);
        let lo_b1: __m256d = _mm512_castpd512_pd256(acc_b_k1);
        let hi_b1: __m256d = _mm512_extractf64x4_pd::<1>(acc_b_k1);
        _mm256_storeu_pd(dst_ptr.add(8), _mm256_sub_pd(lo_a1, hi_a1));
        _mm256_storeu_pd(dst_ptr.add(12), _mm256_add_pd(lo_b1, hi_b1));
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g. `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub unsafe fn reim4_convolution_by_real_const_1coeff_ifma(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64]) {
    use core::arch::x86_64::{__m512d, _mm512_fmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_setzero_pd, _mm512_storeu_pd};

    unsafe {
        let b_size: usize = b.len();

        if k >= a_size + b_size {
            _mm512_storeu_pd(dst.as_mut_ptr(), _mm512_setzero_pd());
            return;
        }

        let j_min: usize = k.saturating_sub(a_size - 1);
        let j_max: usize = (k + 1).min(b_size);

        // Packed accumulator: [Σ ar*br | Σ ai*br]
        let mut acc: __m512d = _mm512_setzero_pd();

        let mut a_ptr: *const f64 = a.as_ptr().add(8 * (k - j_min));
        let mut b_ptr: *const f64 = b.as_ptr().add(j_min);

        for _ in 0..j_max - j_min {
            let a_full: __m512d = _mm512_loadu_pd(a_ptr); // [ar(4) | ai(4)]
            let br: __m512d = _mm512_set1_pd(*b_ptr); // broadcast scalar to all 8 lanes

            acc = _mm512_fmadd_pd(a_full, br, acc); // [ar*br | ai*br]

            a_ptr = a_ptr.sub(8);
            b_ptr = b_ptr.add(1);
        }

        _mm512_storeu_pd(dst.as_mut_ptr(), acc);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g. `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub unsafe fn reim4_convolution_by_real_const_2coeffs_ifma(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64]) {
    use core::arch::x86_64::{__m512d, _mm512_fmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_setzero_pd, _mm512_storeu_pd};

    let b_size: usize = b.len();

    debug_assert!(a.len() >= 8 * a_size);

    let k0: usize = k;
    let k1: usize = k + 1;
    let bound: usize = a_size + b_size;

    if k0 >= bound {
        unsafe {
            let zero: __m512d = _mm512_setzero_pd();
            _mm512_storeu_pd(dst.as_mut_ptr(), zero);
            _mm512_storeu_pd(dst.as_mut_ptr().add(8), zero);
        }
        return;
    }

    unsafe {
        // Packed accumulators: [Σ ar*br | Σ ai*br] per coefficient
        let mut acc_k0: __m512d = _mm512_setzero_pd();
        let mut acc_k1: __m512d = _mm512_setzero_pd();

        let j0_min: usize = (k0 + 1).saturating_sub(a_size);
        let j0_max: usize = (k0 + 1).min(b_size);

        if k1 >= bound {
            let mut a_k0_ptr: *const f64 = a.as_ptr().add(8 * (k0 - j0_min));
            let mut b_ptr: *const f64 = b.as_ptr().add(j0_min);

            for _ in 0..j0_max - j0_min {
                let a0_full: __m512d = _mm512_loadu_pd(a_k0_ptr);
                let br: __m512d = _mm512_set1_pd(*b_ptr);

                acc_k0 = _mm512_fmadd_pd(a0_full, br, acc_k0);

                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
        } else {
            let j1_min: usize = (k1 + 1).saturating_sub(a_size);
            let j1_max: usize = (k1 + 1).min(b_size);

            let mut a_k0_ptr: *const f64 = a.as_ptr().add(8 * (k0 - j0_min));
            let mut a_k1_ptr: *const f64 = a.as_ptr().add(8 * (k1 - j1_min));
            let mut b_ptr: *const f64 = b.as_ptr().add(j0_min);

            // Region 1: k0 only, j ∈ [j0_min, j1_min)
            for _ in 0..j1_min - j0_min {
                let a0_full: __m512d = _mm512_loadu_pd(a_k0_ptr);
                let br: __m512d = _mm512_set1_pd(*b_ptr);

                acc_k0 = _mm512_fmadd_pd(a0_full, br, acc_k0);

                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }

            // Region 2: overlap, contributions to both k0 and k1, j ∈ [j1_min, j0_max)
            for _ in 0..j0_max - j1_min {
                let a0_full: __m512d = _mm512_loadu_pd(a_k0_ptr);
                let a1_full: __m512d = _mm512_loadu_pd(a_k1_ptr);
                let br: __m512d = _mm512_set1_pd(*b_ptr);

                acc_k0 = _mm512_fmadd_pd(a0_full, br, acc_k0);
                acc_k1 = _mm512_fmadd_pd(a1_full, br, acc_k1);

                a_k0_ptr = a_k0_ptr.sub(8);
                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }

            // Region 3: k1 only, j ∈ [j0_max, j1_max)
            for _ in 0..j1_max - j0_max {
                let a1_full: __m512d = _mm512_loadu_pd(a_k1_ptr);
                let br: __m512d = _mm512_set1_pd(*b_ptr);

                acc_k1 = _mm512_fmadd_pd(a1_full, br, acc_k1);

                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
        }

        _mm512_storeu_pd(dst.as_mut_ptr(), acc_k0);
        _mm512_storeu_pd(dst.as_mut_ptr().add(8), acc_k1);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use poulpy_cpu_ref::reference::fft64::reim4::{
        reim4_convolution_1coeff_ref, reim4_convolution_2coeffs_ref, reim4_extract_1blk_from_reim_contiguous_ref,
        reim4_save_1blk_to_reim_contiguous_ref, reim4_vec_mat1col_product_ref, reim4_vec_mat2cols_product_ref,
    };

    use super::*;

    fn reim4_data(size: usize, seed: f64) -> Vec<f64> {
        (0..size * 8).map(|i| (i as f64 * seed + 0.5) / size as f64).collect()
    }

    /// AVX extract+save round-trip matches reference.
    #[test]
    fn reim4_extract_save_1blk_avx_vs_ref() {
        let m = 8usize; // multiple of 4
        let rows = 2usize;
        let blk = 0usize;

        let src: Vec<f64> = (0..2 * rows * m).map(|i| i as f64 + 1.0).collect();
        let mut dst_ifma = vec![0f64; 2 * rows * 4];
        let mut dst_ref = vec![0f64; 2 * rows * 4];

        unsafe { reim4_extract_1blk_from_reim_contiguous_ifma(m, rows, blk, &mut dst_ifma, &src) };
        reim4_extract_1blk_from_reim_contiguous_ref(m, rows, blk, &mut dst_ref, &src);

        assert_eq!(dst_ifma, dst_ref, "reim4_extract_1blk: AVX vs ref mismatch");

        // Also verify save round-trip
        let mut out_ifma = vec![0f64; 2 * rows * m];
        let mut out_ref = vec![0f64; 2 * rows * m];
        unsafe { reim4_save_1blk_to_reim_contiguous_ifma(m, rows, blk, &mut out_ifma, &dst_ifma) };
        reim4_save_1blk_to_reim_contiguous_ref(m, rows, blk, &mut out_ref, &dst_ref);

        assert_eq!(out_ifma, out_ref, "reim4_save_1blk: AVX vs ref mismatch");
    }

    /// AVX `reim4_vec_mat1col_product` matches reference.
    #[test]
    fn reim4_vec_mat1col_product_avx_vs_ref() {
        let nrows = 8usize;
        let u = reim4_data(nrows, 1.3);
        let v = reim4_data(nrows, 2.7);
        let mut dst_ifma = vec![0f64; 8];
        let mut dst_ref = vec![0f64; 8];

        unsafe { reim4_vec_mat1col_product_ifma(nrows, &mut dst_ifma, &u, &v) };
        reim4_vec_mat1col_product_ref(nrows, &mut dst_ref, &u, &v);

        let tol = 1e-12f64;
        for i in 0..8 {
            assert!(
                (dst_ifma[i] - dst_ref[i]).abs() <= tol,
                "mat1col idx={i}: AVX={} ref={}",
                dst_ifma[i],
                dst_ref[i]
            );
        }
    }

    /// AVX `reim4_vec_mat2cols_product` matches reference.
    #[test]
    fn reim4_vec_mat2cols_product_avx_vs_ref() {
        let nrows = 8usize;
        let u = reim4_data(nrows, 1.1);
        let v: Vec<f64> = (0..nrows * 16).map(|i| i as f64 * 0.07 + 0.1).collect();
        let mut dst_ifma = vec![0f64; 16];
        let mut dst_ref = vec![0f64; 16];

        unsafe { reim4_vec_mat2cols_product_ifma(nrows, &mut dst_ifma, &u, &v) };
        reim4_vec_mat2cols_product_ref(nrows, &mut dst_ref, &u, &v);

        let tol = 1e-12f64;
        for i in 0..16 {
            assert!(
                (dst_ifma[i] - dst_ref[i]).abs() <= tol,
                "mat2cols idx={i}: AVX={} ref={}",
                dst_ifma[i],
                dst_ref[i]
            );
        }
    }

    /// AVX `reim4_convolution_1coeff` matches reference for all k values.
    #[test]
    fn reim4_convolution_1coeff_avx_vs_ref() {
        let a_size = 4usize;
        let b_size = 4usize;
        let a = reim4_data(a_size, 1.5);
        let b = reim4_data(b_size, 2.1);

        for k in 0..a_size + b_size + 1 {
            let mut dst_ifma = [0f64; 8];
            let mut dst_ref = [0f64; 8];
            unsafe { reim4_convolution_1coeff_ifma(k, &mut dst_ifma, &a, a_size, &b, b_size) };
            reim4_convolution_1coeff_ref(k, &mut dst_ref, &a, a_size, &b, b_size);
            let tol = 1e-12f64;
            for i in 0..8 {
                assert!(
                    (dst_ifma[i] - dst_ref[i]).abs() <= tol,
                    "conv1coeff k={k} i={i}: AVX={} ref={}",
                    dst_ifma[i],
                    dst_ref[i]
                );
            }
        }
    }

    /// AVX `reim4_convolution_2coeffs` matches reference for all k values.
    #[test]
    fn reim4_convolution_2coeffs_avx_vs_ref() {
        let a_size = 4usize;
        let b_size = 4usize;
        let a = reim4_data(a_size, 1.7);
        let b = reim4_data(b_size, 2.3);

        for k in 0..a_size + b_size + 1 {
            let mut dst_ifma = [0f64; 16];
            let mut dst_ref = [0f64; 16];
            unsafe { reim4_convolution_2coeffs_ifma(k, &mut dst_ifma, &a, a_size, &b, b_size) };
            reim4_convolution_2coeffs_ref(k, &mut dst_ref, &a, a_size, &b, b_size);
            let tol = 1e-12f64;
            for i in 0..16 {
                assert!(
                    (dst_ifma[i] - dst_ref[i]).abs() <= tol,
                    "conv2coeffs k={k} i={i}: AVX={} ref={}",
                    dst_ifma[i],
                    dst_ref[i]
                );
            }
        }
    }
}
