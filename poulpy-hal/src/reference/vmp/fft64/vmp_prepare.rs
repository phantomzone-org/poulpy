use crate::{
    layouts::{Backend, MatZnx, MatZnxToRef, VmpPMatToMut, ZnxInfos, ZnxView, ZnxViewMut},
    reference::{
        reim::{ReimConv, ReimDFTExecute, ReimFFTTable},
        reim4::Reim4Blk,
    },
};

pub fn vmp_prepare_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vmp_prepare<R, A, BE, BLK, CONV, FFT>(table: &ReimFFTTable<f64>, res: &mut R, a: &A, tmp: &mut [f64])
where
    BE: Backend<ScalarPrep = f64>,
    R: VmpPMatToMut<BE>,
    A: MatZnxToRef,
    BLK: Reim4Blk,
    CONV: ReimConv,
    FFT: ReimDFTExecute<ReimFFTTable<f64>, f64>,
{
    let mut res: crate::layouts::VmpPMat<&mut [u8], BE> = res.to_mut();
    let a: MatZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(
            res.cols_in(),
            a.cols_in(),
            "res.cols_in: {} != a.cols_in: {}",
            res.cols_in(),
            a.cols_in()
        );
        assert_eq!(
            res.rows(),
            a.rows(),
            "res.rows: {} != a.rows: {}",
            res.rows(),
            a.rows()
        );
        assert_eq!(
            res.cols_out(),
            a.cols_out(),
            "res.cols_out: {} != a.cols_out: {}",
            res.cols_out(),
            a.cols_out()
        );
        assert_eq!(
            res.size(),
            a.size(),
            "res.size: {} != a.size: {}",
            res.size(),
            a.size()
        );
    }

    let nrows: usize = a.cols_in() * a.rows();
    let ncols: usize = a.cols_out() * a.size();
    vmp_prepare_core::<BLK, CONV, FFT>(table, res.raw_mut(), a.raw(), nrows, ncols, tmp);
}

pub(crate) fn vmp_prepare_core<BLK, CONV, FFT>(
    table: &ReimFFTTable<f64>,
    pmat: &mut [f64],
    mat: &[i64],
    nrows: usize,
    ncols: usize,
    tmp: &mut [f64],
) where
    BLK: Reim4Blk,
    CONV: ReimConv,
    FFT: ReimDFTExecute<ReimFFTTable<f64>, f64>,
{
    let m: usize = table.m();
    let n: usize = m << 1;

    #[cfg(debug_assertions)]
    {
        assert!(n >= 8);
        assert_eq!(mat.len(), n * nrows * ncols);
        assert_eq!(pmat.len(), n * nrows * ncols);
        assert_eq!(tmp.len(), vmp_prepare_tmp_bytes(n) / size_of::<i64>())
    }

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            CONV::reim_from_znx_i64(tmp, &mat[n * (row_i * ncols + col_i)..]);
            FFT::reim_dft_execute(table, tmp);

            let dst: &mut [f64] = if col_i == (ncols - 1) && !ncols.is_multiple_of(2) {
                &mut pmat[col_i * nrows * 8 + row_i * 8..]
            } else {
                &mut pmat[(col_i / 2) * (2 * nrows * 8) + row_i * 2 * 8 + (col_i % 2) * 8..]
            };

            for blk_i in 0..m >> 2 {
                BLK::reim4_extract_1blk_from_reim(m, 1, blk_i, dst, tmp);
            }
        }
    }
}
