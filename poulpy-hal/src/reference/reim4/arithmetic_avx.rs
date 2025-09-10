use crate::reference::reim4::Reim4Blk;

pub struct Reim4BlkAvx;

impl Reim4Blk for Reim4BlkAvx {
    #[inline]
    fn reim4_extract_1blk_from_reim(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        unsafe {
            reim4_extract_1blk_from_reim_avx(m, rows, blk, dst, src);
        }
    }

    #[inline]
    fn reim4_save_1blk_to_reim(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        unsafe {
            reim4_save_1blk_to_reim_avx(m, blk, dst, src);
        }
    }

    #[inline]
    fn reim4_save_2blk_to_reim(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        unsafe {
            reim4_save_2blk_to_reim_avx(m, blk, dst, src);
        }
    }

    #[inline]
    fn reim4_vec_mat1col_product(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        unsafe {
            reim4_vec_mat1col_product_avx(nrows, dst, u, v);
        }
    }

    #[inline]
    fn reim4_vec_mat2cols_product(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        unsafe {
            reim4_vec_mat2cols_product_avx(nrows, dst, u, v);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub fn reim4_extract_1blk_from_reim_avx(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_loadu_pd, _mm256_storeu_pd};

    unsafe {
        let mut src_ptr: *const __m256d = src.as_ptr().add(blk << 2) as *const __m256d; // src + 4*blk
        let mut dst_ptr: *mut __m256d = dst.as_mut_ptr() as *mut __m256d;

        // Each iteration copies 4 doubles; advance src by m doubles each row
        for _ in 0..2 * rows {
            let v = _mm256_loadu_pd(src_ptr as *const f64);
            _mm256_storeu_pd(dst_ptr as *mut f64, v);
            dst_ptr = dst_ptr.add(1);
            src_ptr = src_ptr.add(m / 4);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub fn reim4_save_1blk_to_reim_avx(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    use core::arch::x86_64::{_mm256_loadu_pd, _mm256_storeu_pd};
    unsafe {
        let off: usize = blk * 4;
        let src_ptr: *const f64 = src.as_ptr();
        _mm256_storeu_pd(dst.as_mut_ptr().add(off), _mm256_loadu_pd(src_ptr));
        _mm256_storeu_pd(
            dst.as_mut_ptr().add(m + off),
            _mm256_loadu_pd(src_ptr.add(4)),
        );
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub fn reim4_save_2blk_to_reim_avx(
    m: usize,        //
    blk: usize,      // block index
    dst: &mut [f64], //
    src: &[f64],     // 16 doubles [re1(4), im1(4), re2(4), im2(4)]
) {
    use core::arch::x86_64::{_mm256_loadu_pd, _mm256_storeu_pd};
    unsafe {
        let off: usize = blk * 4;
        let src_ptr: *const f64 = src.as_ptr();
        _mm256_storeu_pd(dst.as_mut_ptr().add(off), _mm256_loadu_pd(src_ptr));
        _mm256_storeu_pd(
            dst.as_mut_ptr().add(off + m),
            _mm256_loadu_pd(src_ptr.add(4)),
        );
        _mm256_storeu_pd(
            dst.as_mut_ptr().add(off + 2 * m),
            _mm256_loadu_pd(src_ptr.add(8)),
        );
        _mm256_storeu_pd(
            dst.as_mut_ptr().add(off + 3 * m),
            _mm256_loadu_pd(src_ptr.add(12)),
        );
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub fn reim4_vec_mat1col_product_avx(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    unsafe {
        use std::arch::x86_64::{_mm256_add_pd, _mm256_sub_pd};

        let mut re1: __m256d = _mm256_setzero_pd();
        let mut im1: __m256d = _mm256_setzero_pd();
        let mut re2: __m256d = _mm256_setzero_pd();
        let mut im2: __m256d = _mm256_setzero_pd();

        for i in 0..nrows {
            let ur: __m256d = _mm256_loadu_pd(u.as_ptr().add(8 * i));
            let ui: __m256d = _mm256_loadu_pd(u.as_ptr().add(8 * i + 4));
            let vr: __m256d = _mm256_loadu_pd(v.as_ptr().add(8 * i));
            let vi: __m256d = _mm256_loadu_pd(v.as_ptr().add(8 * i + 4));

            // re1 = re1 + ur*vr;
            re1 = _mm256_fmadd_pd(ur, vr, re1);
            // im1 = im1 + ur*d;
            im1 = _mm256_fmadd_pd(ur, vi, im1);
            // re2 = re2 + ui*d;
            re2 = _mm256_fmadd_pd(ui, vi, re2);
            // im2 = im2 + ui*vr;
            im2 = _mm256_fmadd_pd(ui, vr, re1);
        }

        // re1 - re2
        _mm256_storeu_pd(dst.as_mut_ptr(), _mm256_sub_pd(re1, re2));

        // im1 + im2
        _mm256_storeu_pd(dst.as_mut_ptr().add(4), _mm256_add_pd(im1, im2));
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub fn reim4_vec_mat2cols_product_avx(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    unsafe {
        let mut re1: __m256d = _mm256_setzero_pd();
        let mut im1: __m256d = _mm256_setzero_pd();
        let mut re2: __m256d = _mm256_setzero_pd();
        let mut im2: __m256d = _mm256_setzero_pd();

        for i in 0..nrows {
            let ur: __m256d = _mm256_loadu_pd(u.as_ptr().add(8 * i));
            let ui: __m256d = _mm256_loadu_pd(u.as_ptr().add(8 * i + 4));

            let ar: __m256d = _mm256_loadu_pd(v.as_ptr().add(16 * i));
            let ai: __m256d = _mm256_loadu_pd(v.as_ptr().add(16 * i + 4));
            let br: __m256d = _mm256_loadu_pd(v.as_ptr().add(16 * i + 8));
            let bi: __m256d = _mm256_loadu_pd(v.as_ptr().add(16 * i + 12));

            // re1 = re1 - ui*ai; re2 = re2 - ui*bi;
            re1 = _mm256_fmsub_pd(ui, ai, re1);
            re2 = _mm256_fmsub_pd(ui, bi, re2);
            // im1 = im1 + ur*ai; im2 = im2 + ur*bi;
            im1 = _mm256_fmadd_pd(ur, ai, im1);
            im2 = _mm256_fmadd_pd(ur, bi, im2);
            // re1 = re1 - ur*ar; re2 = re2 - ur*br;
            re1 = _mm256_fmsub_pd(ur, ar, re1);
            re2 = _mm256_fmsub_pd(ur, br, re2);
            // im1 = im1 + ui*ar; im2 = im2 + ui*br;
            im1 = _mm256_fmadd_pd(ui, ar, im1);
            im2 = _mm256_fmadd_pd(ui, br, im2);
        }

        _mm256_storeu_pd(dst.as_mut_ptr(), re1);
        _mm256_storeu_pd(dst.as_mut_ptr().add(4), im1);
        _mm256_storeu_pd(dst.as_mut_ptr().add(8), re2);
        _mm256_storeu_pd(dst.as_mut_ptr().add(12), im2);
    }
}
