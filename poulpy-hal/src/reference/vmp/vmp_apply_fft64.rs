/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub fn fft64_vmp_apply_dft_to_dft_avx(
    n: usize,
    res: &mut [f64],
    a_dft: &[f64],
    pmat: &[f64],
    nrows: usize,
    ncols: usize,
    tmp_bytes: &mut [f64],
) {
    assert_eq!(pmat.len(), nrows * ncols * n);

    let m: usize = n >> 1;

    let (mat2cols_output, extracted_blk) = tmp_bytes.split_at_mut(16);

    let a_size: usize = a_dft.len();
    let res_size: usize = res.len();

    let row_max: usize = nrows.min(a_size);
    let col_max: usize = ncols.min(res_size);

    for blk_i in 0..m >> 2 {
        let mat_blk_start: &[f64] = &pmat[blk_i * (8 * nrows * ncols)..];

        unsafe {
            reim4_extract_1blk_from_contiguous_reim_avx(m, row_max, blk_i, extracted_blk, a_dft);
        }

        for col_i in (0..col_max - 1).step_by(2) {
            let col_offset: usize = col_i * (8 * nrows);

            unsafe {
                reim4_vec_mat2cols_product_avx2(
                    row_max,
                    mat2cols_output,
                    extracted_blk,
                    &mat_blk_start[col_offset..],
                );
                reim4_save_2blk_to_reim_avx(m, blk_i, &mut res[col_i * n..], mat2cols_output)
            }
        }

        if !col_max.is_multiple_of(2) {
            let last_col: usize = col_max - 1;
            let col_offset: usize = last_col * (8 * nrows);

            if ncols == col_max {
                unsafe {
                    reim4_vec_mat1col_product_avx2(
                        row_max,
                        mat2cols_output,
                        extracted_blk,
                        &mat_blk_start[col_offset..],
                    );
                }
            } else {
                unsafe {
                    reim4_vec_mat2cols_product_avx2(
                        row_max,
                        mat2cols_output,
                        extracted_blk,
                        &mat_blk_start[col_offset..],
                    );
                }
            }
            unsafe {
                reim4_save_1blk_to_reim_avx(m, blk_i, &mut res[last_col * n..], mat2cols_output);
            }
        }
    }

    res[col_max * n..].fill(0f64);
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn reim4_extract_1blk_from_contiguous_reim_avx(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
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

#[allow(dead_code)]
#[inline]
fn reim4_save_1blk_to_reim_scalar(dst: &mut [f64], m: usize, blk: usize, src: &[f64]) {
    let off = blk * 4;
    dst[off..off + 4].copy_from_slice(&src[0..4]);
    dst[off + m..off + m + 4].copy_from_slice(&src[4..8]);
}

#[allow(dead_code)]
#[inline]
fn reim4_save_2blk_to_reim_scalar(dst: &mut [f64], m: usize, blk: usize, src: &[f64]) {
    let off = blk * 4;
    dst[off..off + 4].copy_from_slice(&src[0..4]);
    dst[off + m..off + m + 4].copy_from_slice(&src[4..8]);
    dst[off + 2 * m..off + 2 * m + 4].copy_from_slice(&src[8..12]);
    dst[off + 3 * m..off + 3 * m + 4].copy_from_slice(&src[12..16]);
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn reim4_save_1blk_to_reim_avx(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
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
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn reim4_save_2blk_to_reim_avx(
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

#[allow(dead_code)]
pub fn reim4_vec_mat1cols_product_ref(
    nrows: usize,
    dst: &mut [f64], // 16 doubles: [re1(4), im1(4), re2(4), im2(4)]
    u: &[f64],       // nrows * 8 doubles: [ur(4) | ui(4)] per row
    v: &[f64],       // nrows * 16 doubles: [ar(4) | ai(4) | br(4) | bi(4)] per row
) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(dst.len(), 4, "dst must have 8 doubles");
        assert_eq!(u.len(), nrows * 8, "u must be nrows * 8 doubles");
        assert_eq!(v.len(), nrows * 8, "v must be nrows * 8 doubles");
    }

    // Portable scalar fallback
    let (re, im) = dst.split_at_mut(4);

    // zero accumulators
    re.fill(0f64);
    im.fill(0f64);

    for i in 0..nrows {
        let u_base = 8 * i;
        let ur: &[f64] = &u[u_base..u_base + 4];
        let ui: &[f64] = &u[u_base + 4..u_base + 8];

        let v_base: usize = 8 * i;
        let vr: &[f64] = &v[v_base..v_base + 4];
        let vi: &[f64] = &v[v_base + 4..v_base + 8];

        for k in 0..4 {
            re[k] += ur[k] * vr[k] - ui[k] * vi[k];
            im[k] += ur[k] * vi[k] + ui[k] * vr[k];
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn reim4_vec_mat1col_product_avx2(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
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

#[allow(dead_code)]
pub fn reim4_vec_mat2cols_product_ref(
    nrows: usize,
    dst: &mut [f64], // 16 doubles: [re1(4), im1(4), re2(4), im2(4)]
    u: &[f64],       // nrows * 8 doubles: [ur(4) | ui(4)] per row
    v: &[f64],       // nrows * 16 doubles: [ar(4) | ai(4) | br(4) | bi(4)] per row
) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(dst.len(), 16, "dst must have 16 doubles");
        assert_eq!(u.len(), nrows * 8, "u must be nrows * 8 doubles");
        assert_eq!(v.len(), nrows * 16, "v must be nrows * 16 doubles");
    }

    // Portable scalar fallback
    let (re1, tail) = dst.split_at_mut(4);
    let (im1, tail) = tail.split_at_mut(4);
    let (re2, im2) = tail.split_at_mut(4);

    // zero accumulators
    re1.fill(0f64);
    im1.fill(0f64);
    re2.fill(0f64);
    im2.fill(0f64);

    for i in 0..nrows {
        let u_base = 8 * i;
        let ur: &[f64] = &u[u_base..u_base + 4];
        let ui: &[f64] = &u[u_base + 4..u_base + 8];

        let v_base: usize = 16 * i;
        let ar: &[f64] = &v[v_base..v_base + 4];
        let ai: &[f64] = &v[v_base + 4..v_base + 8];
        let br: &[f64] = &v[v_base + 8..v_base + 12];
        let bi: &[f64] = &v[v_base + 12..v_base + 16];

        for k in 0..4 {
            // re1 -= ui * ai; re2 -= ui * bi;
            re1[k] -= ui[k] * ai[k];
            re2[k] -= ui[k] * bi[k];
            // im1 += ur * ai; im2 += ur * bi;
            im1[k] += ur[k] * ai[k];
            im2[k] += ur[k] * bi[k];
            // re1 -= ur * ar; re2 -= ur * br;
            re1[k] -= ur[k] * ar[k];
            re2[k] -= ur[k] * br[k];
            // im1 += ui * ar; im2 += ui * br;
            im1[k] += ui[k] * ar[k];
            im2[k] += ui[k] * br[k];
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn reim4_vec_mat2cols_product_avx2(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
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
