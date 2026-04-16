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

use std::arch::x86_64::{
    __m128d, __m256d, __m512d, __m512i, _mm_loadu_pd, _mm256_loadu_pd, _mm256_set_m128d, _mm256_storeu_pd, _mm512_add_pd,
    _mm512_castpd256_pd512, _mm512_castpd512_pd256, _mm512_extractf64x4_pd, _mm512_fmadd_pd, _mm512_fmsub_pd, _mm512_insertf64x4,
    _mm512_loadu_pd, _mm512_mul_pd, _mm512_permutex2var_pd, _mm512_set_epi64, _mm512_set1_pd, _mm512_shuffle_pd,
    _mm512_storeu_pd, _mm512_sub_pd, _mm512_unpackhi_pd, _mm512_unpacklo_pd,
};

use crate::fft64::reim::as_arr;

#[target_feature(enable = "avx512f")]
pub(crate) fn ifft_avx512(m: usize, omg: &[f64], data: &mut [f64]) {
    // m <= 16 falls through to the reference implementation: it is too small
    // for the AVX-512 base case (`ifft16x2_avx512` processes 2 blocks per
    // call, so it needs m >= 32). For m >= 32, the BFS dispatcher always
    // produces an even number of IFFT16 blocks and uses `ifft16x2_avx512`.
    if m <= 16 {
        use poulpy_cpu_ref::reference::fft64::reim::ifft_ref;
        ifft_ref(m, omg, data);
        return;
    }

    assert!(data.len() == 2 * m);
    let (re, im) = data.split_at_mut(m);

    if m <= 2048 {
        ifft_bfs_16_avx512(m, re, im, omg, 0);
    } else {
        ifft_rec_16_avx512(m, re, im, omg, 0);
    }
}

#[target_feature(enable = "avx512f")]
fn ifft_rec_16_avx512(m: usize, re: &mut [f64], im: &mut [f64], omg: &[f64], mut pos: usize) -> usize {
    if m <= 2048 {
        return ifft_bfs_16_avx512(m, re, im, omg, pos);
    };
    let h: usize = m >> 1;
    pos = ifft_rec_16_avx512(h, re, im, omg, pos);
    pos = ifft_rec_16_avx512(h, &mut re[h..], &mut im[h..], omg, pos);
    inv_twiddle_ifft_avx512(h, re, im, *as_arr::<2, f64>(&omg[pos..]));
    pos += 2;
    pos
}

#[target_feature(enable = "avx512f")]
fn ifft_bfs_16_avx512(m: usize, re: &mut [f64], im: &mut [f64], omg: &[f64], mut pos: usize) -> usize {
    let log_m: usize = (usize::BITS - (m - 1).leading_zeros()) as usize;

    // m is always a multiple of 32 here (smallest BFS input is m == 32);
    // ifft16x2 processes 2 blocks per call.
    let mut off = 0;
    while off + 32 <= m {
        unsafe {
            ifft16x2_avx512(&mut re[off..off + 32], &mut im[off..off + 32], &omg[pos..pos + 32]);
        }
        pos += 32;
        off += 32;
    }

    let mut h: usize = 16;
    let m_half: usize = m >> 1;

    while h < m_half {
        let mm: usize = h << 2;
        for off in (0..m).step_by(mm) {
            inv_bitwiddle_ifft_avx512(h, &mut re[off..], &mut im[off..], as_arr::<4, f64>(&omg[pos..]));
            pos += 4;
        }
        h = mm;
    }

    if !log_m.is_multiple_of(2) {
        inv_twiddle_ifft_avx512(h, re, im, *as_arr::<2, f64>(&omg[pos..]));
        pos += 2;
    }

    pos
}

#[target_feature(enable = "avx512f")]
fn inv_twiddle_ifft_avx512(h: usize, re: &mut [f64], im: &mut [f64], omg: [f64; 2]) {
    unsafe {
        let omr: __m512d = _mm512_set1_pd(omg[0]);
        let omi: __m512d = _mm512_set1_pd(omg[1]);
        let mut r0: *mut f64 = re.as_mut_ptr();
        let mut r1: *mut f64 = re.as_mut_ptr().add(h);
        let mut i0: *mut f64 = im.as_mut_ptr();
        let mut i1: *mut f64 = im.as_mut_ptr().add(h);
        for _ in (0..h).step_by(8) {
            let mut ur0: __m512d = _mm512_loadu_pd(r0);
            let mut ur1: __m512d = _mm512_loadu_pd(r1);
            let mut ui0: __m512d = _mm512_loadu_pd(i0);
            let mut ui1: __m512d = _mm512_loadu_pd(i1);
            let tra = _mm512_sub_pd(ur0, ur1);
            let tia = _mm512_sub_pd(ui0, ui1);
            ur0 = _mm512_add_pd(ur0, ur1);
            ui0 = _mm512_add_pd(ui0, ui1);
            ur1 = _mm512_mul_pd(omi, tia);
            ui1 = _mm512_mul_pd(omi, tra);
            ur1 = _mm512_fmsub_pd(omr, tra, ur1);
            ui1 = _mm512_fmadd_pd(omr, tia, ui1);
            _mm512_storeu_pd(r0, ur0);
            _mm512_storeu_pd(r1, ur1);
            _mm512_storeu_pd(i0, ui0);
            _mm512_storeu_pd(i1, ui1);

            r0 = r0.add(8);
            r1 = r1.add(8);
            i0 = i0.add(8);
            i1 = i1.add(8);
        }
    }
}

#[target_feature(enable = "avx512f")]
fn inv_bitwiddle_ifft_avx512(h: usize, re: &mut [f64], im: &mut [f64], omg: &[f64; 4]) {
    unsafe {
        let mut r0: *mut f64 = re.as_mut_ptr();
        let mut r1: *mut f64 = re.as_mut_ptr().add(h);
        let mut r2: *mut f64 = re.as_mut_ptr().add(2 * h);
        let mut r3: *mut f64 = re.as_mut_ptr().add(3 * h);
        let mut i0: *mut f64 = im.as_mut_ptr();
        let mut i1: *mut f64 = im.as_mut_ptr().add(h);
        let mut i2: *mut f64 = im.as_mut_ptr().add(2 * h);
        let mut i3: *mut f64 = im.as_mut_ptr().add(3 * h);
        let omar: __m512d = _mm512_set1_pd(omg[0]);
        let omai: __m512d = _mm512_set1_pd(omg[1]);
        let ombr: __m512d = _mm512_set1_pd(omg[2]);
        let ombi: __m512d = _mm512_set1_pd(omg[3]);
        for _ in (0..h).step_by(8) {
            let mut ur0: __m512d = _mm512_loadu_pd(r0);
            let mut ur1: __m512d = _mm512_loadu_pd(r1);
            let mut ur2: __m512d = _mm512_loadu_pd(r2);
            let mut ur3: __m512d = _mm512_loadu_pd(r3);
            let mut ui0: __m512d = _mm512_loadu_pd(i0);
            let mut ui1: __m512d = _mm512_loadu_pd(i1);
            let mut ui2: __m512d = _mm512_loadu_pd(i2);
            let mut ui3: __m512d = _mm512_loadu_pd(i3);

            let mut tra: __m512d = _mm512_sub_pd(ur0, ur1);
            let mut trb: __m512d = _mm512_sub_pd(ur2, ur3);
            let mut tia: __m512d = _mm512_sub_pd(ui0, ui1);
            let mut tib: __m512d = _mm512_sub_pd(ui2, ui3);
            ur0 = _mm512_add_pd(ur0, ur1);
            ur2 = _mm512_add_pd(ur2, ur3);
            ui0 = _mm512_add_pd(ui0, ui1);
            ui2 = _mm512_add_pd(ui2, ui3);
            ur1 = _mm512_mul_pd(omai, tia);
            ur3 = _mm512_mul_pd(omar, tib);
            ui1 = _mm512_mul_pd(omai, tra);
            ui3 = _mm512_mul_pd(omar, trb);
            ur1 = _mm512_fmsub_pd(omar, tra, ur1);
            ur3 = _mm512_fmadd_pd(omai, trb, ur3);
            ui1 = _mm512_fmadd_pd(omar, tia, ui1);
            ui3 = _mm512_fmsub_pd(omai, tib, ui3);

            tra = _mm512_sub_pd(ur0, ur2);
            trb = _mm512_sub_pd(ur1, ur3);
            tia = _mm512_sub_pd(ui0, ui2);
            tib = _mm512_sub_pd(ui1, ui3);
            ur0 = _mm512_add_pd(ur0, ur2);
            ur1 = _mm512_add_pd(ur1, ur3);
            ui0 = _mm512_add_pd(ui0, ui2);
            ui1 = _mm512_add_pd(ui1, ui3);
            ur2 = _mm512_mul_pd(ombi, tia);
            ur3 = _mm512_mul_pd(ombi, tib);
            ui2 = _mm512_mul_pd(ombi, tra);
            ui3 = _mm512_mul_pd(ombi, trb);
            ur2 = _mm512_fmsub_pd(ombr, tra, ur2);
            ur3 = _mm512_fmsub_pd(ombr, trb, ur3);
            ui2 = _mm512_fmadd_pd(ombr, tia, ui2);
            ui3 = _mm512_fmadd_pd(ombr, tib, ui3);

            _mm512_storeu_pd(r0, ur0);
            _mm512_storeu_pd(r1, ur1);
            _mm512_storeu_pd(r2, ur2);
            _mm512_storeu_pd(r3, ur3);
            _mm512_storeu_pd(i0, ui0);
            _mm512_storeu_pd(i1, ui1);
            _mm512_storeu_pd(i2, ui2);
            _mm512_storeu_pd(i3, ui3);

            r0 = r0.add(8);
            r1 = r1.add(8);
            r2 = r2.add(8);
            r3 = r3.add(8);
            i0 = i0.add(8);
            i1 = i1.add(8);
            i2 = i2.add(8);
            i3 = i3.add(8);
        }
    }
}

/// Process two consecutive IFFT16 blocks in parallel using __m512d (AVX-512F).
///
/// `re`, `im` must each be at least 32 doubles long. `omg` must be at least 32 doubles long.
/// Block A is at offsets `[0..16]`, block B is at offsets `[16..32]`.
#[target_feature(enable = "avx512f")]
unsafe fn ifft16x2_avx512(re: &mut [f64], im: &mut [f64], omg: &[f64]) {
    #[inline(always)]
    unsafe fn load_pair(p: *const f64, off: usize) -> __m512d {
        unsafe {
            let a: __m256d = _mm256_loadu_pd(p.add(off));
            let b: __m256d = _mm256_loadu_pd(p.add(off + 16));
            _mm512_insertf64x4::<1>(_mm512_castpd256_pd512(a), b)
        }
    }

    #[inline(always)]
    unsafe fn store_pair(p: *mut f64, off: usize, v: __m512d) {
        unsafe {
            let lo: __m256d = _mm512_castpd512_pd256(v);
            let hi: __m256d = _mm512_extractf64x4_pd::<1>(v);
            _mm256_storeu_pd(p.add(off), lo);
            _mm256_storeu_pd(p.add(off + 16), hi);
        }
    }

    #[inline(always)]
    unsafe fn load_narrow_twiddle(omg_ptr: *const f64, off_a: usize, off_b: usize) -> (__m512d, __m512d) {
        unsafe {
            let omx_a: __m128d = _mm_loadu_pd(omg_ptr.add(off_a));
            let omx_b: __m128d = _mm_loadu_pd(omg_ptr.add(off_b));
            let om_a256: __m256d = _mm256_set_m128d(omx_a, omx_a);
            let om_b256: __m256d = _mm256_set_m128d(omx_b, omx_b);
            let om_riri: __m512d = _mm512_insertf64x4::<1>(_mm512_castpd256_pd512(om_a256), om_b256);
            let omi: __m512d = _mm512_unpackhi_pd(om_riri, om_riri);
            let omr: __m512d = _mm512_unpacklo_pd(om_riri, om_riri);
            (omr, omi)
        }
    }

    #[inline(always)]
    unsafe fn load_wide_twiddle(omg_ptr: *const f64, off_a: usize, off_b: usize) -> (__m512d, __m512d) {
        unsafe {
            let om_a: __m256d = _mm256_loadu_pd(omg_ptr.add(off_a));
            let om_b: __m256d = _mm256_loadu_pd(omg_ptr.add(off_b));
            let om_full: __m512d = _mm512_insertf64x4::<1>(_mm512_castpd256_pd512(om_a), om_b);
            let omi: __m512d = _mm512_shuffle_pd::<0xFF>(om_full, om_full);
            let omr: __m512d = _mm512_shuffle_pd::<0x00>(om_full, om_full);
            (omr, omi)
        }
    }

    unsafe {
        let re_ptr: *mut f64 = re.as_mut_ptr();
        let im_ptr: *mut f64 = im.as_mut_ptr();
        let omg_ptr: *const f64 = omg.as_ptr();

        let perm_high: __m512i = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
        let perm_low: __m512i = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);

        // stage 0: load inputs
        let mut ra0: __m512d = load_pair(re_ptr, 0);
        let mut ra1: __m512d = load_pair(re_ptr, 4);
        let mut ra2: __m512d = load_pair(re_ptr, 8);
        let mut ra3: __m512d = load_pair(re_ptr, 12);
        let mut ia0: __m512d = load_pair(im_ptr, 0);
        let mut ia1: __m512d = load_pair(im_ptr, 4);
        let mut ia2: __m512d = load_pair(im_ptr, 8);
        let mut ia3: __m512d = load_pair(im_ptr, 12);

        // ───────────────── stage 1 ─────────────────
        // stage-4-style load: omr at off 0/16, omi at off 4/20
        let s1_omr: __m512d = {
            let a: __m256d = _mm256_loadu_pd(omg_ptr.add(0));
            let b: __m256d = _mm256_loadu_pd(omg_ptr.add(16));
            _mm512_insertf64x4::<1>(_mm512_castpd256_pd512(a), b)
        };
        let s1_omi: __m512d = {
            let a: __m256d = _mm256_loadu_pd(omg_ptr.add(4));
            let b: __m256d = _mm256_loadu_pd(omg_ptr.add(20));
            _mm512_insertf64x4::<1>(_mm512_castpd256_pd512(a), b)
        };

        // vperm2f128 $0x31, ymm2, ymm0, ymm8 => t8 = high|high (ra0, ra2)
        // vperm2f128 $0x31, ymm3, ymm1, ymm9 => t9 = high|high (ra1, ra3)
        // vperm2f128 $0x31, ymm6, ymm4, ymm10 => t10 = high|high (ia0, ia2)
        // vperm2f128 $0x31, ymm7, ymm5, ymm11 => t11 = high|high (ia1, ia3)
        let mut t8: __m512d = _mm512_permutex2var_pd(ra0, perm_high, ra2);
        let mut t9: __m512d = _mm512_permutex2var_pd(ra1, perm_high, ra3);
        let mut t10: __m512d = _mm512_permutex2var_pd(ia0, perm_high, ia2);
        let mut t11: __m512d = _mm512_permutex2var_pd(ia1, perm_high, ia3);
        // vperm2f128 $0x20, ymm2, ymm0, ymm0 => ra0 = low|low (ra0, ra2)
        // vperm2f128 $0x20, ymm3, ymm1, ymm1 => ra1 = low|low (ra1, ra3)
        // vperm2f128 $0x20, ymm6, ymm4, ymm2 => ra2 = low|low (ia0, ia2)
        // vperm2f128 $0x20, ymm7, ymm5, ymm3 => ra3 = low|low (ia1, ia3)
        let new_ra0: __m512d = _mm512_permutex2var_pd(ra0, perm_low, ra2);
        let new_ra1: __m512d = _mm512_permutex2var_pd(ra1, perm_low, ra3);
        let new_ra2: __m512d = _mm512_permutex2var_pd(ia0, perm_low, ia2);
        let new_ra3: __m512d = _mm512_permutex2var_pd(ia1, perm_low, ia3);
        ra0 = new_ra0;
        ra1 = new_ra1;
        ra2 = new_ra2;
        ra3 = new_ra3;

        // vunpckhpd %ymm1,%ymm0,%ymm4 => ia0 = unpackhi(ra0, ra1)
        // vunpckhpd %ymm3,%ymm2,%ymm6 => ia2 = unpackhi(ra2, ra3)
        // vunpckhpd %ymm9,%ymm8,%ymm5 => ia1 = unpackhi(t8, t9)
        // vunpckhpd %ymm11,%ymm10,%ymm7 => ia3 = unpackhi(t10, t11)
        let new_ia0: __m512d = _mm512_unpackhi_pd(ra0, ra1);
        let new_ia2: __m512d = _mm512_unpackhi_pd(ra2, ra3);
        let new_ia1: __m512d = _mm512_unpackhi_pd(t8, t9);
        let new_ia3: __m512d = _mm512_unpackhi_pd(t10, t11);
        // vunpcklpd %ymm1,%ymm0,%ymm0 => ra0 = unpacklo(ra0, ra1)
        // vunpcklpd %ymm3,%ymm2,%ymm2 => ra2 = unpacklo(ra2, ra3)
        // vunpcklpd %ymm9,%ymm8,%ymm1 => ra1 = unpacklo(t8, t9)
        // vunpcklpd %ymm11,%ymm10,%ymm3 => ra3 = unpacklo(t10, t11)
        let new_ra0: __m512d = _mm512_unpacklo_pd(ra0, ra1);
        let new_ra2: __m512d = _mm512_unpacklo_pd(ra2, ra3);
        let new_ra1: __m512d = _mm512_unpacklo_pd(t8, t9);
        let new_ra3: __m512d = _mm512_unpacklo_pd(t10, t11);
        ia0 = new_ia0;
        ia2 = new_ia2;
        ia1 = new_ia1;
        ia3 = new_ia3;
        ra0 = new_ra0;
        ra2 = new_ra2;
        ra1 = new_ra1;
        ra3 = new_ra3;

        // vsubpd %ymm4,%ymm0,%ymm8  => t8  = ra0 - ia0
        // vsubpd %ymm5,%ymm1,%ymm9  => t9  = ra1 - ia1
        // vsubpd %ymm6,%ymm2,%ymm10 => t10 = ra2 - ia2
        // vsubpd %ymm7,%ymm3,%ymm11 => t11 = ra3 - ia3
        t8 = _mm512_sub_pd(ra0, ia0);
        t9 = _mm512_sub_pd(ra1, ia1);
        t10 = _mm512_sub_pd(ra2, ia2);
        t11 = _mm512_sub_pd(ra3, ia3);
        // vaddpd %ymm4,%ymm0,%ymm0  => ra0 += ia0
        // vaddpd %ymm5,%ymm1,%ymm1  => ra1 += ia1
        // vaddpd %ymm6,%ymm2,%ymm2  => ra2 += ia2
        // vaddpd %ymm7,%ymm3,%ymm3  => ra3 += ia3
        ra0 = _mm512_add_pd(ra0, ia0);
        ra1 = _mm512_add_pd(ra1, ia1);
        ra2 = _mm512_add_pd(ra2, ia2);
        ra3 = _mm512_add_pd(ra3, ia3);
        // vmulpd %ymm10,%ymm13,%ymm4 => ia0 = t10 * omi
        // vmulpd %ymm11,%ymm12,%ymm5 => ia1 = t11 * omr
        // vmulpd %ymm8,%ymm13,%ymm6  => ia2 = t8 * omi
        // vmulpd %ymm9,%ymm12,%ymm7  => ia3 = t9 * omr
        ia0 = _mm512_mul_pd(t10, s1_omi);
        ia1 = _mm512_mul_pd(t11, s1_omr);
        ia2 = _mm512_mul_pd(t8, s1_omi);
        ia3 = _mm512_mul_pd(t9, s1_omr);
        // vfmsub231pd %ymm8,%ymm12,%ymm4  => ia0 = t8  * omr - ia0
        // vfmadd231pd %ymm9,%ymm13,%ymm5  => ia1 = t9  * omi + ia1
        // vfmadd231pd %ymm10,%ymm12,%ymm6 => ia2 = t10 * omr + ia2
        // vfmsub231pd %ymm11,%ymm13,%ymm7 => ia3 = t11 * omi - ia3
        ia0 = _mm512_fmsub_pd(t8, s1_omr, ia0);
        ia1 = _mm512_fmadd_pd(t9, s1_omi, ia1);
        ia2 = _mm512_fmadd_pd(t10, s1_omr, ia2);
        ia3 = _mm512_fmsub_pd(t11, s1_omi, ia3);

        // vunpckhpd %ymm7,%ymm3,%ymm11 => t11 = unpackhi(ra3, ia3)
        // vunpckhpd %ymm5,%ymm1,%ymm9  => t9  = unpackhi(ra1, ia1)
        // vunpcklpd %ymm7,%ymm3,%ymm10 => t10 = unpacklo(ra3, ia3)
        // vunpcklpd %ymm5,%ymm1,%ymm8  => t8  = unpacklo(ra1, ia1)
        t11 = _mm512_unpackhi_pd(ra3, ia3);
        t9 = _mm512_unpackhi_pd(ra1, ia1);
        t10 = _mm512_unpacklo_pd(ra3, ia3);
        t8 = _mm512_unpacklo_pd(ra1, ia1);
        // vunpckhpd %ymm6,%ymm2,%ymm3 => ra3 = unpackhi(ra2, ia2)
        // vunpckhpd %ymm4,%ymm0,%ymm1 => ra1 = unpackhi(ra0, ia0)
        // vunpcklpd %ymm6,%ymm2,%ymm2 => ra2 = unpacklo(ra2, ia2)
        // vunpcklpd %ymm4,%ymm0,%ymm0 => ra0 = unpacklo(ra0, ia0)
        let new_ra3: __m512d = _mm512_unpackhi_pd(ra2, ia2);
        let new_ra1: __m512d = _mm512_unpackhi_pd(ra0, ia0);
        let new_ra2: __m512d = _mm512_unpacklo_pd(ra2, ia2);
        let new_ra0: __m512d = _mm512_unpacklo_pd(ra0, ia0);
        ra3 = new_ra3;
        ra1 = new_ra1;
        ra2 = new_ra2;
        ra0 = new_ra0;

        // ───────────────── stage 2 ─────────────────
        // At this point:
        //   regs ra0..ra3 hold "0,1,2,3" and t8,t9,t10,t11 hold "8,9,10,11"
        //   (matching ymm0..ymm3, ymm8..ymm11 in the assembly)
        let (s2_omr, s2_omi) = load_wide_twiddle(omg_ptr, 8, 24);

        // vsubpd %ymm8,%ymm0,%ymm4  => ia0 = ra0 - t8
        // vsubpd %ymm9,%ymm1,%ymm5  => ia1 = ra1 - t9
        // vsubpd %ymm10,%ymm2,%ymm6 => ia2 = ra2 - t10
        // vsubpd %ymm11,%ymm3,%ymm7 => ia3 = ra3 - t11
        ia0 = _mm512_sub_pd(ra0, t8);
        ia1 = _mm512_sub_pd(ra1, t9);
        ia2 = _mm512_sub_pd(ra2, t10);
        ia3 = _mm512_sub_pd(ra3, t11);
        // vaddpd %ymm8,%ymm0,%ymm0  => ra0 += t8
        // vaddpd %ymm9,%ymm1,%ymm1  => ra1 += t9
        // vaddpd %ymm10,%ymm2,%ymm2 => ra2 += t10
        // vaddpd %ymm11,%ymm3,%ymm3 => ra3 += t11
        ra0 = _mm512_add_pd(ra0, t8);
        ra1 = _mm512_add_pd(ra1, t9);
        ra2 = _mm512_add_pd(ra2, t10);
        ra3 = _mm512_add_pd(ra3, t11);
        // vmulpd %ymm6,%ymm13,%ymm8  => t8  = ia2 * omi
        // vmulpd %ymm7,%ymm12,%ymm9  => t9  = ia3 * omr
        // vmulpd %ymm4,%ymm13,%ymm10 => t10 = ia0 * omi
        // vmulpd %ymm5,%ymm12,%ymm11 => t11 = ia1 * omr
        t8 = _mm512_mul_pd(ia2, s2_omi);
        t9 = _mm512_mul_pd(ia3, s2_omr);
        t10 = _mm512_mul_pd(ia0, s2_omi);
        t11 = _mm512_mul_pd(ia1, s2_omr);
        // vfmsub231pd %ymm4,%ymm12,%ymm8  => t8  = ia0 * omr - t8
        // vfmadd231pd %ymm5,%ymm13,%ymm9  => t9  = ia1 * omi + t9
        // vfmadd231pd %ymm6,%ymm12,%ymm10 => t10 = ia2 * omr + t10
        // vfmsub231pd %ymm7,%ymm13,%ymm11 => t11 = ia3 * omi - t11
        t8 = _mm512_fmsub_pd(ia0, s2_omr, t8);
        t9 = _mm512_fmadd_pd(ia1, s2_omi, t9);
        t10 = _mm512_fmadd_pd(ia2, s2_omr, t10);
        t11 = _mm512_fmsub_pd(ia3, s2_omi, t11);

        // vperm2f128 $0x31, ymm10, ymm2, ymm6 => ia2 = high|high (ra2, t10)
        // vperm2f128 $0x31, ymm11, ymm3, ymm7 => ia3 = high|high (ra3, t11)
        // vperm2f128 $0x20, ymm10, ymm2, ymm4 => ia0 = low|low  (ra2, t10)
        // vperm2f128 $0x20, ymm11, ymm3, ymm5 => ia1 = low|low  (ra3, t11)
        let new_ia2: __m512d = _mm512_permutex2var_pd(ra2, perm_high, t10);
        let new_ia3: __m512d = _mm512_permutex2var_pd(ra3, perm_high, t11);
        let new_ia0: __m512d = _mm512_permutex2var_pd(ra2, perm_low, t10);
        let new_ia1: __m512d = _mm512_permutex2var_pd(ra3, perm_low, t11);
        // vperm2f128 $0x31, ymm8, ymm0, ymm2 => ra2 = high|high (ra0, t8)
        // vperm2f128 $0x31, ymm9, ymm1, ymm3 => ra3 = high|high (ra1, t9)
        // vperm2f128 $0x20, ymm8, ymm0, ymm0 => ra0 = low|low  (ra0, t8)
        // vperm2f128 $0x20, ymm9, ymm1, ymm1 => ra1 = low|low  (ra1, t9)
        let new_ra2: __m512d = _mm512_permutex2var_pd(ra0, perm_high, t8);
        let new_ra3: __m512d = _mm512_permutex2var_pd(ra1, perm_high, t9);
        let new_ra0: __m512d = _mm512_permutex2var_pd(ra0, perm_low, t8);
        let new_ra1: __m512d = _mm512_permutex2var_pd(ra1, perm_low, t9);
        ia2 = new_ia2;
        ia3 = new_ia3;
        ia0 = new_ia0;
        ia1 = new_ia1;
        ra2 = new_ra2;
        ra3 = new_ra3;
        ra0 = new_ra0;
        ra1 = new_ra1;

        // ───────────────── stage 3 ─────────────────
        let (s3_omr, s3_omi) = load_narrow_twiddle(omg_ptr, 12, 28);

        // At this point regs are: ra0..ra3, ia0..ia3 matching ymm0..ymm3, ymm4..ymm7.
        // vsubpd %ymm1,%ymm0,%ymm8  => t8  = ra0 - ra1
        // vsubpd %ymm3,%ymm2,%ymm9  => t9  = ra2 - ra3
        // vsubpd %ymm5,%ymm4,%ymm10 => t10 = ia0 - ia1
        // vsubpd %ymm7,%ymm6,%ymm11 => t11 = ia2 - ia3
        t8 = _mm512_sub_pd(ra0, ra1);
        t9 = _mm512_sub_pd(ra2, ra3);
        t10 = _mm512_sub_pd(ia0, ia1);
        t11 = _mm512_sub_pd(ia2, ia3);
        // vaddpd %ymm1,%ymm0,%ymm0 => ra0 += ra1
        // vaddpd %ymm3,%ymm2,%ymm2 => ra2 += ra3
        // vaddpd %ymm5,%ymm4,%ymm4 => ia0 += ia1
        // vaddpd %ymm7,%ymm6,%ymm6 => ia2 += ia3
        ra0 = _mm512_add_pd(ra0, ra1);
        ra2 = _mm512_add_pd(ra2, ra3);
        ia0 = _mm512_add_pd(ia0, ia1);
        ia2 = _mm512_add_pd(ia2, ia3);
        // vmulpd %ymm10,%ymm13,%ymm1 => ra1 = t10 * omi
        // vmulpd %ymm11,%ymm12,%ymm3 => ra3 = t11 * omr
        // vmulpd %ymm8,%ymm13,%ymm5  => ia1 = t8 * omi
        // vmulpd %ymm9,%ymm12,%ymm7  => ia3 = t9 * omr
        ra1 = _mm512_mul_pd(t10, s3_omi);
        ra3 = _mm512_mul_pd(t11, s3_omr);
        ia1 = _mm512_mul_pd(t8, s3_omi);
        ia3 = _mm512_mul_pd(t9, s3_omr);
        // vfmsub231pd %ymm8,%ymm12,%ymm1  => ra1 = t8  * omr - ra1
        // vfmadd231pd %ymm9,%ymm13,%ymm3  => ra3 = t9  * omi + ra3
        // vfmadd231pd %ymm10,%ymm12,%ymm5 => ia1 = t10 * omr + ia1
        // vfmsub231pd %ymm11,%ymm13,%ymm7 => ia3 = t11 * omi - ia3
        ra1 = _mm512_fmsub_pd(t8, s3_omr, ra1);
        ra3 = _mm512_fmadd_pd(t9, s3_omi, ra3);
        ia1 = _mm512_fmadd_pd(t10, s3_omr, ia1);
        ia3 = _mm512_fmsub_pd(t11, s3_omi, ia3);

        // ───────────────── stage 4 ─────────────────
        let (s4_omr, s4_omi) = load_narrow_twiddle(omg_ptr, 14, 30);

        // vsubpd %ymm2,%ymm0,%ymm8  => t8  = ra0 - ra2
        // vsubpd %ymm3,%ymm1,%ymm9  => t9  = ra1 - ra3
        // vsubpd %ymm6,%ymm4,%ymm10 => t10 = ia0 - ia2
        // vsubpd %ymm7,%ymm5,%ymm11 => t11 = ia1 - ia3
        t8 = _mm512_sub_pd(ra0, ra2);
        t9 = _mm512_sub_pd(ra1, ra3);
        t10 = _mm512_sub_pd(ia0, ia2);
        t11 = _mm512_sub_pd(ia1, ia3);
        // vaddpd %ymm2,%ymm0,%ymm0 => ra0 += ra2
        // vaddpd %ymm3,%ymm1,%ymm1 => ra1 += ra3
        // vaddpd %ymm6,%ymm4,%ymm4 => ia0 += ia2
        // vaddpd %ymm7,%ymm5,%ymm5 => ia1 += ia3
        ra0 = _mm512_add_pd(ra0, ra2);
        ra1 = _mm512_add_pd(ra1, ra3);
        ia0 = _mm512_add_pd(ia0, ia2);
        ia1 = _mm512_add_pd(ia1, ia3);
        // vmulpd %ymm10,%ymm13,%ymm2 => ra2 = t10 * omi
        // vmulpd %ymm11,%ymm13,%ymm3 => ra3 = t11 * omi
        // vmulpd %ymm8,%ymm13,%ymm6  => ia2 = t8  * omi
        // vmulpd %ymm9,%ymm13,%ymm7  => ia3 = t9  * omi
        ra2 = _mm512_mul_pd(t10, s4_omi);
        ra3 = _mm512_mul_pd(t11, s4_omi);
        ia2 = _mm512_mul_pd(t8, s4_omi);
        ia3 = _mm512_mul_pd(t9, s4_omi);
        // vfmsub231pd %ymm8,%ymm12,%ymm2  => ra2 = t8  * omr - ra2
        // vfmsub231pd %ymm9,%ymm12,%ymm3  => ra3 = t9  * omr - ra3
        // vfmadd231pd %ymm10,%ymm12,%ymm6 => ia2 = t10 * omr + ia2
        // vfmadd231pd %ymm11,%ymm12,%ymm7 => ia3 = t11 * omr + ia3
        ra2 = _mm512_fmsub_pd(t8, s4_omr, ra2);
        ra3 = _mm512_fmsub_pd(t9, s4_omr, ra3);
        ia2 = _mm512_fmadd_pd(t10, s4_omr, ia2);
        ia3 = _mm512_fmadd_pd(t11, s4_omr, ia3);

        // stores
        store_pair(re_ptr, 0, ra0);
        store_pair(re_ptr, 4, ra1);
        store_pair(re_ptr, 8, ra2);
        store_pair(re_ptr, 12, ra3);
        store_pair(im_ptr, 0, ia0);
        store_pair(im_ptr, 4, ia1);
        store_pair(im_ptr, 8, ia2);
        store_pair(im_ptr, 12, ia3);
    }
}

#[test]
fn test_ifft_avx512() {
    use super::*;

    #[target_feature(enable = "avx512f")]
    fn internal(log_m: usize) {
        use poulpy_cpu_ref::reference::fft64::reim::ReimIFFTRef;

        let m: usize = 1 << log_m;

        let table: ReimIFFTTable<f64> = ReimIFFTTable::<f64>::new(m);

        let mut values_0: Vec<f64> = vec![0f64; m << 1];
        let scale: f64 = 1.0f64 / m as f64;
        values_0.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as f64 * scale);

        let mut values_1: Vec<f64> = vec![0f64; m << 1];
        values_1.iter_mut().zip(values_0.iter()).for_each(|(y, x)| *y = *x);

        ReimIFFTIfma::reim_dft_execute(&table, &mut values_0);
        ReimIFFTRef::reim_dft_execute(&table, &mut values_1);

        let max_diff: f64 = 1.0 / ((1u64 << (53 - log_m - 1)) as f64);

        for i in 0..m * 2 {
            let diff: f64 = (values_0[i] - values_1[i]).abs();
            assert!(diff <= max_diff, "{} -> {}-{} = {}", i, values_0[i], values_1[i], diff)
        }
    }

    for log_m in 0..16 {
        unsafe { internal(log_m) }
    }
}
