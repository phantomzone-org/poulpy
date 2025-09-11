use crate::reference::{reim::as_arr, reim4::Reim4Blk};

pub struct Reim4BlkRef;

impl Reim4Blk for Reim4BlkRef {
    #[inline(always)]
    fn reim4_extract_1blk_from_reim(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_extract_1blk_from_reim_ref(m, rows, blk, dst, src);
    }

    #[inline(always)]
    fn reim4_save_1blk_to_reim<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_save_1blk_to_reim_ref::<OVERWRITE>(m, blk, dst, src);
    }

    #[inline(always)]
    fn reim4_save_2blk_to_reim<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        reim4_save_2blk_to_reim_ref::<OVERWRITE>(m, blk, dst, src);
    }

    #[inline(always)]
    fn reim4_vec_mat1col_product(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        reim4_vec_mat1col_product_ref(nrows, dst, u, v);
    }

    #[inline(always)]
    fn reim4_vec_mat2cols_product(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        reim4_vec_mat2cols_product_ref(nrows, dst, u, v);
    }

    #[inline(always)]
    fn reim4_vec_mat2cols_2ndcol_product(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        reim4_vec_mat2cols_2ndcol_product(nrows, dst, u, v);
    }
}

#[inline(always)]
fn reim4_extract_1blk_from_reim_ref(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    let mut offset: usize = blk << 2;

    debug_assert!(blk < (m >> 2));
    debug_assert!(dst.len() >= 2 * rows * 4);

    for chunk in dst.chunks_exact_mut(4).take(2 * rows) {
        chunk.copy_from_slice(&src[offset..offset + 4]);
        offset += m
    }
}

#[inline(always)]
fn reim4_save_1blk_to_reim_ref<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    let mut offset: usize = blk << 2;

    debug_assert!(blk < (m >> 2));
    debug_assert!(dst.len() >= offset + m + 4);
    debug_assert!(src.len() >= 8);

    let dst_off = &mut dst[offset..offset + 4];

    if OVERWRITE {
        dst_off.copy_from_slice(&src[0..4]);
    } else {
        dst_off[0] += src[0];
        dst_off[1] += src[1];
        dst_off[2] += src[2];
        dst_off[3] += src[3];
    }

    offset += m;

    let dst_off = &mut dst[offset..offset + 4];
    if OVERWRITE {
        dst_off.copy_from_slice(&src[4..8]);
    } else {
        dst_off[0] += src[4];
        dst_off[1] += src[5];
        dst_off[2] += src[6];
        dst_off[3] += src[7];
    }
}

#[inline(always)]
fn reim4_save_2blk_to_reim_ref<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    let mut offset: usize = blk << 2;

    debug_assert!(blk < (m >> 2));
    debug_assert!(dst.len() >= offset + 3 * m + 4);
    debug_assert!(src.len() >= 16);

    let dst_off = &mut dst[offset..offset + 4];
    if OVERWRITE {
        dst_off.copy_from_slice(&src[0..4]);
    } else {
        dst_off[0] += src[0];
        dst_off[1] += src[1];
        dst_off[2] += src[2];
        dst_off[3] += src[3];
    }

    offset += m;
    let dst_off = &mut dst[offset..offset + 4];
    if OVERWRITE {
        dst_off.copy_from_slice(&src[4..8]);
    } else {
        dst_off[0] += src[4];
        dst_off[1] += src[5];
        dst_off[2] += src[6];
        dst_off[3] += src[7];
    }

    offset += m;

    let dst_off = &mut dst[offset..offset + 4];
    if OVERWRITE {
        dst_off.copy_from_slice(&src[8..12]);
    } else {
        dst_off[0] += src[8];
        dst_off[1] += src[9];
        dst_off[2] += src[10];
        dst_off[3] += src[11];
    }

    offset += m;
    let dst_off = &mut dst[offset..offset + 4];
    if OVERWRITE {
        dst_off.copy_from_slice(&src[12..16]);
    } else {
        dst_off[0] += src[12];
        dst_off[1] += src[13];
        dst_off[2] += src[14];
        dst_off[3] += src[15];
    }
}

#[inline(always)]
fn reim4_vec_mat1col_product_ref(
    nrows: usize,
    dst: &mut [f64], // 8 doubles: [re1(4), im1(4)]
    u: &[f64],       // nrows * 8 doubles: [ur(4) | ui(4)] per row
    v: &[f64],       // nrows * 8 doubles: [ar(4) | ai(4)] per row
) {
    #[cfg(debug_assertions)]
    {
        assert!(dst.len() >= 8, "dst must have at least 8 doubles");
        assert!(u.len() >= nrows * 8, "u must be at least nrows * 8 doubles");
        assert!(v.len() >= nrows * 8, "v must be at least nrows * 8 doubles");
    }
    let mut acc: [f64; 8] = [0f64; 8];
    let mut j = 0;
    for _ in 0..nrows {
        reim4_add_mul(&mut acc, as_arr(&u[j..]), as_arr(&v[j..]));
        j += 8;
    }
    dst[0..8].copy_from_slice(&acc);
}

#[inline(always)]
fn reim4_vec_mat2cols_2ndcol_product(
    nrows: usize,
    dst: &mut [f64], // 8 doubles: [re1(4), im1(4), re2(4), im2(4)]
    u: &[f64],       // nrows * 8 doubles: [ur(4) | ui(4)] per row
    v: &[f64],       // nrows * 16 doubles: [x | x | br(4) | bi(4)] per row
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
    let mut acc: [f64; 8] = [0f64; 8];
    for i in 0..nrows {
        let _1j: usize = i << 3;
        let _2j: usize = i << 4;
        reim4_add_mul(&mut acc, as_arr(&u[_1j..]), as_arr(&v[_2j + 8..]));
    }
    dst[0..8].copy_from_slice(&acc);
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
    let mut acc_0: [f64; 8] = [0f64; 8];
    let mut acc_1: [f64; 8] = [0f64; 8];
    for i in 0..nrows {
        let _1j: usize = i << 3;
        let _2j: usize = i << 4;
        let u_j: &[f64; 8] = as_arr(&u[_1j..]);
        reim4_add_mul(&mut acc_0, u_j, as_arr(&v[_2j..]));
        reim4_add_mul(&mut acc_1, u_j, as_arr(&v[_2j + 8..]));
    }
    dst[0..8].copy_from_slice(&acc_0);
    dst[8..16].copy_from_slice(&acc_1);
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
