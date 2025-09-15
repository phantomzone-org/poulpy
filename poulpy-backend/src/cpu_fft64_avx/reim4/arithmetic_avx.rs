/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub fn reim4_extract_1blk_from_reim_avx(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
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
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub fn reim4_save_1blk_to_reim_avx<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
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
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub fn reim4_save_2blk_to_reim_avx<const OVERWRITE: bool>(
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
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub fn reim4_vec_mat1col_product_avx(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    #[cfg(debug_assertions)]
    {
        assert!(dst.len() >= 8, "dst must have at least 8 doubles");
        assert!(u.len() >= nrows * 8, "u must be at least nrows * 8 doubles");
        assert!(v.len() >= nrows * 8, "v must be at least nrows * 8 doubles");
    }

    unsafe {
        use std::arch::x86_64::{_mm256_add_pd, _mm256_sub_pd};

        let mut re1: __m256d = _mm256_setzero_pd();
        let mut im1: __m256d = _mm256_setzero_pd();
        let mut re2: __m256d = _mm256_setzero_pd();
        let mut im2: __m256d = _mm256_setzero_pd();

        let mut u_ptr: *const f64 = u.as_ptr();
        let mut v_ptr: *const f64 = v.as_ptr();

        for _ in 0..nrows {
            let ur: __m256d = _mm256_loadu_pd(u_ptr);
            let ui: __m256d = _mm256_loadu_pd(u_ptr.add(4));
            let vr: __m256d = _mm256_loadu_pd(v_ptr);
            let vi: __m256d = _mm256_loadu_pd(v_ptr.add(4));

            // re1 = re1 + ur*vr;
            re1 = _mm256_fmadd_pd(ur, vr, re1);
            // im1 = im1 + ur*d;
            im1 = _mm256_fmadd_pd(ur, vi, im1);
            // re2 = re2 + ui*d;
            re2 = _mm256_fmadd_pd(ui, vi, re2);
            // im2 = im2 + ui*vr;
            im2 = _mm256_fmadd_pd(ui, vr, im2);

            u_ptr = u_ptr.add(8);
            v_ptr = v_ptr.add(8);
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

    #[cfg(debug_assertions)]
    {
        assert!(
            dst.len() >= 8,
            "dst must be at least 8 doubles but is {}",
            dst.len()
        );
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
        let mut re1: __m256d = _mm256_setzero_pd();
        let mut im1: __m256d = _mm256_setzero_pd();
        let mut re2: __m256d = _mm256_setzero_pd();
        let mut im2: __m256d = _mm256_setzero_pd();

        let mut u_ptr: *const f64 = u.as_ptr();
        let mut v_ptr: *const f64 = v.as_ptr();

        for _ in 0..nrows {
            let ur: __m256d = _mm256_loadu_pd(u_ptr);
            let ui: __m256d = _mm256_loadu_pd(u_ptr.add(4));

            let ar: __m256d = _mm256_loadu_pd(v_ptr);
            let ai: __m256d = _mm256_loadu_pd(v_ptr.add(4));
            let br: __m256d = _mm256_loadu_pd(v_ptr.add(8));
            let bi: __m256d = _mm256_loadu_pd(v_ptr.add(12));

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

            u_ptr = u_ptr.add(8);
            v_ptr = v_ptr.add(16);
        }

        _mm256_storeu_pd(dst.as_mut_ptr(), re1);
        _mm256_storeu_pd(dst.as_mut_ptr().add(4), im1);
        _mm256_storeu_pd(dst.as_mut_ptr().add(8), re2);
        _mm256_storeu_pd(dst.as_mut_ptr().add(12), im2);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub fn reim4_vec_mat2cols_2ndcol_product_avx(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    #[cfg(debug_assertions)]
    {
        assert_eq!(dst.len(), 16, "dst must have 16 doubles");
        assert!(u.len() >= nrows * 8, "u must be at least nrows * 8 doubles");
        assert!(
            v.len() >= nrows * 16,
            "v must be at least nrows * 16 doubles"
        );
    }

    unsafe {
        let mut re1: __m256d = _mm256_setzero_pd();
        let mut im1: __m256d = _mm256_setzero_pd();

        let mut u_ptr: *const f64 = u.as_ptr();
        let mut v_ptr: *const f64 = v.as_ptr().add(8); // Offset to 2nd column

        for _ in 0..nrows {
            let ur: __m256d = _mm256_loadu_pd(u_ptr);
            let ui: __m256d = _mm256_loadu_pd(u_ptr.add(4));

            let ar: __m256d = _mm256_loadu_pd(v_ptr);
            let ai: __m256d = _mm256_loadu_pd(v_ptr.add(4));

            // re1 = re1 - ui*ai; re2 = re2 - ui*bi;
            re1 = _mm256_fmsub_pd(ui, ai, re1);
            // im1 = im1 + ur*ai; im2 = im2 + ur*bi;
            im1 = _mm256_fmadd_pd(ur, ai, im1);
            // re1 = re1 - ur*ar; re2 = re2 - ur*br;
            re1 = _mm256_fmsub_pd(ur, ar, re1);
            // im1 = im1 + ui*ar; im2 = im2 + ui*br;
            im1 = _mm256_fmadd_pd(ui, ar, im1);

            u_ptr = u_ptr.add(8);
            v_ptr = v_ptr.add(16);
        }

        _mm256_storeu_pd(dst.as_mut_ptr(), re1);
        _mm256_storeu_pd(dst.as_mut_ptr().add(4), im1);
    }
}
