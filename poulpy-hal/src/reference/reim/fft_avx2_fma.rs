use std::arch::x86_64::{
    __m128d, __m256d, _mm_load_pd, _mm256_add_pd, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_loadu_pd, _mm256_mul_pd,
    _mm256_permute2f128_pd, _mm256_set_m128d, _mm256_storeu_pd, _mm256_sub_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd,
};

use crate::reference::reim::{as_arr, as_arr_mut};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) fn fft_avx2_fma(m: usize, omg: &[f64], data: &mut [f64]) {
    if m < 16 {
        use crate::reference::reim::fft_ref::fft_ref;
        fft_ref(m, omg, data);
        return;
    }

    assert!(data.len() == 2 * m);
    let (re, im) = data.split_at_mut(m);

    if m == 16 {
        fft16_avx2_fma(
            as_arr_mut::<16, f64>(re),
            as_arr_mut::<16, f64>(im),
            as_arr::<16, f64>(omg),
        )
    } else if m <= 2048 {
        fft_bfs_16_avx2_fma(m, re, im, omg, 0);
    } else {
        fft_rec_16_avx2_fma(m, re, im, omg, 0);
    }
}

unsafe extern "sysv64" {
    unsafe fn fft16_avx2_fma_asm(re: *mut f64, im: *mut f64, omg: *const f64);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
fn fft16_avx2_fma(re: &mut [f64; 16], im: &mut [f64; 16], omg: &[f64; 16]) {
    unsafe {
        fft16_avx2_fma_asm(re.as_mut_ptr(), im.as_mut_ptr(), omg.as_ptr());
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
fn fft_rec_16_avx2_fma(m: usize, re: &mut [f64], im: &mut [f64], omg: &[f64], mut pos: usize) -> usize {
    if m <= 2048 {
        return fft_bfs_16_avx2_fma(m, re, im, omg, pos);
    };

    let h: usize = m >> 1;
    twiddle_fft_avx2_fma(h, re, im, *as_arr::<2, f64>(&omg[pos..]));
    pos += 2;
    pos = fft_rec_16_avx2_fma(h, re, im, omg, pos);
    pos = fft_rec_16_avx2_fma(h, &mut re[h..], &mut im[h..], omg, pos);
    pos
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
fn fft_bfs_16_avx2_fma(m: usize, re: &mut [f64], im: &mut [f64], omg: &[f64], mut pos: usize) -> usize {
    let log_m: usize = (usize::BITS - (m - 1).leading_zeros()) as usize;
    let mut mm: usize = m;

    if !log_m.is_multiple_of(2) {
        let h: usize = mm >> 1;
        twiddle_fft_avx2_fma(h, re, im, *as_arr::<2, f64>(&omg[pos..]));
        pos += 2;
        mm = h
    }

    while mm > 16 {
        let h: usize = mm >> 2;
        for off in (0..m).step_by(mm) {
            bitwiddle_fft_avx2_fma(
                h,
                &mut re[off..],
                &mut im[off..],
                as_arr::<4, f64>(&omg[pos..]),
            );

            pos += 4;
        }
        mm = h
    }

    for off in (0..m).step_by(16) {
        fft16_avx2_fma(
            as_arr_mut::<16, f64>(&mut re[off..]),
            as_arr_mut::<16, f64>(&mut im[off..]),
            as_arr::<16, f64>(&omg[pos..]),
        );

        pos += 16;
    }

    pos
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
fn twiddle_fft_avx2_fma(h: usize, re: &mut [f64], im: &mut [f64], omg: [f64; 2]) {
    unsafe {
        let omx: __m128d = _mm_load_pd(omg.as_ptr());
        let omra: __m256d = _mm256_set_m128d(omx, omx);
        let omi: __m256d = _mm256_unpackhi_pd(omra, omra);
        let omr: __m256d = _mm256_unpacklo_pd(omra, omra);
        let mut r0: *mut f64 = re.as_mut_ptr();
        let mut r1: *mut f64 = re.as_mut_ptr().add(h);
        let mut i0: *mut f64 = im.as_mut_ptr();
        let mut i1: *mut f64 = im.as_mut_ptr().add(h);

        for _ in (0..h).step_by(4) {
            let mut ur0: __m256d = _mm256_loadu_pd(r0);
            let mut ur1: __m256d = _mm256_loadu_pd(r1);
            let mut ui0: __m256d = _mm256_loadu_pd(i0);
            let mut ui1: __m256d = _mm256_loadu_pd(i1);
            let mut tra: __m256d = _mm256_mul_pd(omi, ui1);
            let mut tia: __m256d = _mm256_mul_pd(omi, ur1);

            tra = _mm256_fmsub_pd(omr, ur1, tra);
            tia = _mm256_fmadd_pd(omr, ui1, tia);
            ur1 = _mm256_sub_pd(ur0, tra);
            ui1 = _mm256_sub_pd(ui0, tia);
            ur0 = _mm256_add_pd(ur0, tra);
            ui0 = _mm256_add_pd(ui0, tia);

            _mm256_storeu_pd(r0, ur0);
            _mm256_storeu_pd(r1, ur1);
            _mm256_storeu_pd(i0, ui0);
            _mm256_storeu_pd(i1, ui1);

            r0 = r0.add(4);
            r1 = r1.add(4);
            i0 = i0.add(4);
            i1 = i1.add(4);
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
fn bitwiddle_fft_avx2_fma(h: usize, re: &mut [f64], im: &mut [f64], omg: &[f64; 4]) {
    unsafe {
        let mut r0: *mut f64 = re.as_mut_ptr();
        let mut r1: *mut f64 = re.as_mut_ptr().add(h);
        let mut r2: *mut f64 = re.as_mut_ptr().add(2 * h);
        let mut r3: *mut f64 = re.as_mut_ptr().add(3 * h);
        let mut i0: *mut f64 = im.as_mut_ptr();
        let mut i1: *mut f64 = im.as_mut_ptr().add(h);
        let mut i2: *mut f64 = im.as_mut_ptr().add(2 * h);
        let mut i3: *mut f64 = im.as_mut_ptr().add(3 * h);
        let om0: __m256d = _mm256_loadu_pd(omg.as_ptr());
        let omb: __m256d = _mm256_permute2f128_pd(om0, om0, 0x11);
        let oma: __m256d = _mm256_permute2f128_pd(om0, om0, 0x00);
        let omai: __m256d = _mm256_unpackhi_pd(oma, oma);
        let omar: __m256d = _mm256_unpacklo_pd(oma, oma);
        let ombi: __m256d = _mm256_unpackhi_pd(omb, omb);
        let ombr: __m256d = _mm256_unpacklo_pd(omb, omb);
        for _ in (0..h).step_by(4) {
            let mut ur0: __m256d = _mm256_loadu_pd(r0);
            let mut ur1: __m256d = _mm256_loadu_pd(r1);
            let mut ur2: __m256d = _mm256_loadu_pd(r2);
            let mut ur3: __m256d = _mm256_loadu_pd(r3);
            let mut ui0: __m256d = _mm256_loadu_pd(i0);
            let mut ui1: __m256d = _mm256_loadu_pd(i1);
            let mut ui2: __m256d = _mm256_loadu_pd(i2);
            let mut ui3: __m256d = _mm256_loadu_pd(i3);

            let mut tra: __m256d = _mm256_mul_pd(omai, ui2);
            let mut trb: __m256d = _mm256_mul_pd(omai, ui3);
            let mut tia: __m256d = _mm256_mul_pd(omai, ur2);
            let mut tib: __m256d = _mm256_mul_pd(omai, ur3);
            tra = _mm256_fmsub_pd(omar, ur2, tra);
            trb = _mm256_fmsub_pd(omar, ur3, trb);
            tia = _mm256_fmadd_pd(omar, ui2, tia);
            tib = _mm256_fmadd_pd(omar, ui3, tib);
            ur2 = _mm256_sub_pd(ur0, tra);
            ur3 = _mm256_sub_pd(ur1, trb);
            ui2 = _mm256_sub_pd(ui0, tia);
            ui3 = _mm256_sub_pd(ui1, tib);
            ur0 = _mm256_add_pd(ur0, tra);
            ur1 = _mm256_add_pd(ur1, trb);
            ui0 = _mm256_add_pd(ui0, tia);
            ui1 = _mm256_add_pd(ui1, tib);

            tra = _mm256_mul_pd(ombi, ui1);
            trb = _mm256_mul_pd(ombr, ui3);
            tia = _mm256_mul_pd(ombi, ur1);
            tib = _mm256_mul_pd(ombr, ur3);
            tra = _mm256_fmsub_pd(ombr, ur1, tra);
            trb = _mm256_fmadd_pd(ombi, ur3, trb);
            tia = _mm256_fmadd_pd(ombr, ui1, tia);
            tib = _mm256_fmsub_pd(ombi, ui3, tib);
            ur1 = _mm256_sub_pd(ur0, tra);
            ur3 = _mm256_add_pd(ur2, trb);
            ui1 = _mm256_sub_pd(ui0, tia);
            ui3 = _mm256_add_pd(ui2, tib);
            ur0 = _mm256_add_pd(ur0, tra);
            ur2 = _mm256_sub_pd(ur2, trb);
            ui0 = _mm256_add_pd(ui0, tia);
            ui2 = _mm256_sub_pd(ui2, tib);

            _mm256_storeu_pd(r0, ur0);
            _mm256_storeu_pd(r1, ur1);
            _mm256_storeu_pd(r2, ur2);
            _mm256_storeu_pd(r3, ur3);
            _mm256_storeu_pd(i0, ui0);
            _mm256_storeu_pd(i1, ui1);
            _mm256_storeu_pd(i2, ui2);
            _mm256_storeu_pd(i3, ui3);

            r0 = r0.add(4);
            r1 = r1.add(4);
            r2 = r2.add(4);
            r3 = r3.add(4);
            i0 = i0.add(4);
            i1 = i1.add(4);
            i2 = i2.add(4);
            i3 = i3.add(4);
        }
    }
}

/// # Safety: this function is safe.
#[target_feature(enable = "avx2")]
fn print_m256d_as_f64x4(v: __m256d) -> String {
    let mut a: [f64; 4] = [0f64; 4];
    unsafe {
        _mm256_storeu_pd(a.as_mut_ptr(), v);
    }
    format!("{a:12.8?}")
}

#[test]
fn test_fft_avx2_fma() {
    use super::*;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "avx2,fma")]
    fn internal(log_m: usize) {
        let m = 1 << log_m;

        let table: TableFFT<f64> = TableFFT::<f64>::new(m);

        let mut values_0: Vec<f64> = vec![0f64; m << 1];
        let scale: f64 = 1.0f64 / m as f64;
        values_0
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = (i + 1) as f64 * scale);

        let mut values_1: Vec<f64> = vec![0f64; m << 1];
        values_1
            .iter_mut()
            .zip(values_0.iter())
            .for_each(|(y, x)| *y = *x);

        table.execute_avx2_fma(&mut values_0);
        table.execute(&mut values_1);

        let max_diff: f64 = 1.0 / ((1u64 << (53 - log_m - 1)) as f64);

        for i in 0..m * 2 {
            let diff: f64 = (values_0[i] - values_1[i]).abs();
            assert!(
                diff <= max_diff,
                "{} -> {}-{} = {}",
                i,
                values_0[i],
                values_1[i],
                diff
            )
        }
    }

    if std::is_x86_feature_detected!("avx2") {
        for log_m in 0..16 {
            unsafe { internal(log_m) }
        }
    } else {
        eprintln!("skipping: CPU lacks avx2");
    }
}
