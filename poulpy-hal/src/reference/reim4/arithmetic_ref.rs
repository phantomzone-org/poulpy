use crate::reference::reim4::Reim4Blk;

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
pub fn reim4_extract_1blk_from_reim_ref(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    debug_assert!(blk < (m >> 2));
    debug_assert!(dst.len() >= 2 * rows * 4);

    for (k, chunk) in dst.chunks_exact_mut(4).take(2 * rows).enumerate() {
        let start = k * m;
        chunk.copy_from_slice(&src[start..start + 4]);
    }
}

#[inline]
pub fn reim4_save_1blk_to_reim_ref(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    let off: usize = blk * 4;

    // debug-only bounds checks
    debug_assert!(dst.len() >= off + m + 4);
    debug_assert!(src.len() >= 8);

    for (k, chunk) in src.chunks_exact(4).take(2).enumerate() {
        let start = off + k * m;
        dst[start..start + 4].copy_from_slice(chunk);
    }
}

#[inline]
pub fn reim4_save_2blk_to_reim_ref(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    let off: usize = blk * 4;

    // debug-only bounds checks
    debug_assert!(dst.len() >= off + 3 * m + 4);
    debug_assert!(src.len() >= 16);

    for (k, chunk) in src.chunks_exact(4).take(4).enumerate() {
        let start = off + k * m;
        dst[start..start + 4].copy_from_slice(chunk);
    }
}

#[inline]
pub fn reim4_vec_mat1col_product_ref(
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

pub fn reim4_vec_mat2cols_product_ref(
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
