use crate::{
    api::{
        DFT, IDFTTmpA, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{MatZnx, Module, ScratchOwned, VecZnx, VecZnxBig, VecZnxDft, VmpPMat, ZnxInfos, ZnxViewMut},
    oep::{
        DFTImpl, IDFTTmpAImpl, ModuleNewImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, VecZnxBigAllocImpl,
        VecZnxBigNormalizeImpl, VecZnxBigNormalizeTmpBytesImpl, VecZnxDftAllocImpl, VmpApplyDftToDftImpl,
        VmpApplyDftToDftTmpBytesImpl, VmpPMatAllocImpl, VmpPMatPrepareImpl,
    },
};

use crate::layouts::Backend;

pub fn test_vmp_apply<B>()
where
    B: Backend
        + ModuleNewImpl<B>
        + VmpApplyDftToDftTmpBytesImpl<B>
        + VecZnxBigNormalizeTmpBytesImpl<B>
        + VmpPMatAllocImpl<B>
        + VecZnxDftAllocImpl<B>
        + VecZnxBigAllocImpl<B>
        + VmpPMatPrepareImpl<B>
        + DFTImpl<B>
        + VmpApplyDftToDftImpl<B>
        + IDFTTmpAImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + VecZnxBigNormalizeImpl<B>,
{
    let log_n: i32 = 5;
    let n: usize = 1 << log_n;

    let module: Module<B> = Module::<B>::new(n as u64);
    let basek: usize = 15;
    let a_size: usize = 5;
    let mat_size: usize = 6;
    let res_size: usize = a_size;

    [1, 2].iter().for_each(|cols_in| {
        [1, 2].iter().for_each(|cols_out| {
            let a_cols: usize = *cols_in;
            let res_cols: usize = *cols_out;

            let mat_rows: usize = a_size;
            let mat_cols_in: usize = a_cols;
            let mat_cols_out: usize = res_cols;

            let mut scratch = ScratchOwned::alloc(
                module.vmp_apply_dft_to_dft_tmp_bytes(
                    res_size,
                    a_size,
                    mat_rows,
                    mat_cols_in,
                    mat_cols_out,
                    mat_size,
                ) | module.vec_znx_big_normalize_tmp_bytes(),
            );

            let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, a_cols, a_size);

            (0..a_cols).for_each(|i| {
                a.at_mut(i, a_size - 1)[i + 1] = 1;
            });

            let mut vmp: VmpPMat<Vec<u8>, B> = module.vmp_pmat_alloc(mat_rows, mat_cols_in, mat_cols_out, mat_size);

            let mut c_dft: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(mat_cols_out, mat_size);
            let mut c_big: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(mat_cols_out, mat_size);

            let mut mat: MatZnx<Vec<u8>> = MatZnx::alloc(n, mat_rows, mat_cols_in, mat_cols_out, mat_size);

            // Construts a [VecZnxMatDft] that performs cyclic rotations on each submatrix.
            (0..a.size()).for_each(|row_i| {
                (0..mat_cols_in).for_each(|col_in_i| {
                    (0..mat_cols_out).for_each(|col_out_i| {
                        let idx = 1 + col_in_i * mat_cols_out + col_out_i;
                        mat.at_mut(row_i, col_in_i).at_mut(col_out_i, row_i)[idx] = 1_i64; // X^{idx}
                    });
                });
            });

            module.vmp_prepare(&mut vmp, &mat, scratch.borrow());

            let mut a_dft: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(a_cols, a_size);
            (0..a_cols).for_each(|i| {
                module.dft(1, 0, &mut a_dft, i, &a, i);
            });

            module.vmp_apply_dft_to_dft(&mut c_dft, &a_dft, &vmp, scratch.borrow());

            let mut res_have_vi64: Vec<i64> = vec![i64::default(); n];

            let mut res_have: VecZnx<Vec<u8>> = VecZnx::alloc(n, res_cols, res_size);
            (0..mat_cols_out).for_each(|i| {
                module.idft_tmp_a(&mut c_big, i, &mut c_dft, i);
                module.vec_znx_big_normalize(basek, &mut res_have, i, &c_big, i, scratch.borrow());
            });

            (0..mat_cols_out).for_each(|col_i| {
                let mut res_want_vi64: Vec<i64> = vec![i64::default(); n];
                (0..a_cols).for_each(|i| {
                    res_want_vi64[(i + 1) + (1 + i * mat_cols_out + col_i)] = 1;
                });
                res_have.decode_vec_i64(basek, col_i, basek * a_size, &mut res_have_vi64);
                assert_eq!(res_have_vi64, res_want_vi64);
            });
        });
    });
}
