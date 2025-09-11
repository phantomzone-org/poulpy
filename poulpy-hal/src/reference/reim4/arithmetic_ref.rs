use crate::reference::{
    reim::{as_arr, as_arr_mut},
    reim4::Reim4Blk,
};

pub struct Reim4BlkRef;

impl Reim4Blk for Reim4BlkRef {
    #[inline]
    fn reim4_extract_1blk_from_reim(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_extract_1blk_from_reim_ref(m, rows, blk, dst, src);
    }

    #[inline]
    fn reim4_save_1blk_to_reim(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_save_1blk_to_reim_ref(m, blk, dst, src);
    }

    #[inline]
    fn reim4_save_2blk_to_reim(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_save_2blk_to_reim_ref(m, blk, dst, src);
    }

    #[inline]
    fn reim4_vec_mat1col_product(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        reim4_vec_mat1col_product_ref(nrows, dst, u, v);
    }

    #[inline]
    fn reim4_vec_mat2cols_product(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        reim4_vec_mat2cols_product_ref(nrows, dst, u, v);
    }
}

#[inline]
fn reim4_extract_1blk_from_reim_ref(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    let mut offset: usize = blk << 2;

    debug_assert!(blk < (m >> 2));
    debug_assert!(dst.len() >= 2 * rows * 4);

    for chunk in dst.chunks_exact_mut(4).take(2 * rows) {
        chunk.copy_from_slice(&src[offset..offset + 4]);
        offset += m
    }
}

#[inline]
fn reim4_save_1blk_to_reim_ref(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    let mut offset: usize = blk << 2;

    debug_assert!(blk < (m >> 2));
    debug_assert!(dst.len() >= offset + m + 4);
    debug_assert!(src.len() >= 8);

    for chunk in src.chunks_exact(4).take(2) {
        dst[offset..offset + 4].copy_from_slice(chunk);
        offset += m
    }
}

#[inline]
fn reim4_save_2blk_to_reim_ref(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    let mut offset: usize = blk << 2;

    debug_assert!(blk < (m >> 2));
    debug_assert!(dst.len() >= offset + 3 * m + 4);
    debug_assert!(src.len() >= 16);

    for chunk in src.chunks_exact(4).take(4) {
        dst[offset..offset + 4].copy_from_slice(chunk);
        offset += m
    }
}

#[inline]
fn reim4_vec_mat1col_product_ref(
    nrows: usize,
    dst: &mut [f64], // 16 doubles: [re1(4), im1(4), re2(4), im2(4)]
    u: &[f64],       // nrows * 8 doubles: [ur(4) | ui(4)] per row
    v: &[f64],       // nrows * 16 doubles: [ar(4) | ai(4) | br(4) | bi(4)] per row
) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(dst.len(), 4, "dst must have 8 doubles");
        assert!(u.len() >= nrows * 8, "u must be at least nrows * 8 doubles");
        assert!(v.len() >= nrows * 8, "v must be at least nrows * 8 doubles");
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

#[inline(always)]
fn reim4_vec_mat2cols_product_ref(
    nrows: usize,
    dst: &mut [f64], // 16 doubles: [re1(4), im1(4), re2(4), im2(4)]
    u: &[f64],       // nrows * 8 doubles: [ur(4) | ui(4)] per row
    v: &[f64],       // nrows * 16 doubles: [ar(4) | ai(4) | br(4) | bi(4)] per row
) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(dst.len(), 16, "dst must have 16 doubles");
        assert!(u.len() >= nrows * 8, "u must be at least nrows * 8 doubles");
        assert!(
            v.len() >= nrows * 16,
            "v must be at least nrows * 16 doubles"
        );
    }

    // zero accumulators
    dst[..16].fill(0f64);

    for i in 0..nrows {
        let _1j: usize = i << 3;
        let _2j: usize = i << 4;
        let u_j: &[f64; 8] = as_arr(&u[_1j..]);
        reim4_add_mul(as_arr_mut(&mut dst[0..]), u_j, as_arr(&v[_2j..]));
        reim4_add_mul(as_arr_mut(&mut dst[8..]), u_j, as_arr(&v[_2j + 8..]));
    }
}

#[inline(always)]
fn reim4_add_mul(dst: &mut [f64; 8], a: &[f64; 8], b: &[f64; 8]) {
    for k in 0..4 {
        let ar: f64 = a[k];
        let br: f64 = b[k];
        let ai: f64 = a[k + 4];
        let bi: f64 = b[k + 4];
        dst[k] += ar * br - ai * bi;
        dst[k + 4] += ar * bi + ai * br;
    }
}
