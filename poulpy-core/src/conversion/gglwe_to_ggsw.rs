use poulpy_hal::{
    api::{
        ScratchAvailable, ScratchTakeBasic, VecZnxBigAddSmallInplace, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalize,
    },
    layouts::{Backend, DataMut, Module, Scratch, VecZnx, VecZnxBig, VecZnxDft, VecZnxDftToRef, VecZnxToRef},
};

use crate::{
    GGLWEProduct, GLWECopy, ScratchTakeCore,
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToGGSWKeyPrepared, GGLWEToGGSWKeyPreparedToRef, GGLWEToRef, GGSW, GGSWInfos, GGSWToMut, GLWE,
        GLWEInfos, LWEInfos,
    },
};

impl GGLWE<Vec<u8>> {
    pub fn from_gglw_tmp_bytes<R, A, M, BE: Backend>(module: &M, res_infos: &R, tsk_infos: &A) -> usize
    where
        M: GGSWFromGGLWE<BE>,
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        module.ggsw_from_gglwe_tmp_bytes(res_infos, tsk_infos)
    }
}

impl<D: DataMut> GGSW<D> {
    pub fn from_gglwe<G, M, T, BE: Backend>(&mut self, module: &M, gglwe: &G, tsk: &T, scratch: &mut Scratch<BE>)
    where
        M: GGSWFromGGLWE<BE>,
        G: GGLWEToRef,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.ggsw_from_gglwe(self, gglwe, tsk, scratch);
    }
}

impl<BE: Backend> GGSWFromGGLWE<BE> for Module<BE>
where
    Self: GGSWExpandRows<BE> + GLWECopy,
{
    fn ggsw_from_gglwe_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        self.ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos)
    }

    fn ggsw_from_gglwe<R, A, T>(&self, res: &mut R, a: &A, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGLWEToRef,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGLWE<&[u8]> = &a.to_ref();
        let tsk: &GGLWEToGGSWKeyPrepared<&[u8], BE> = &tsk.to_ref();

        assert_eq!(res.rank(), a.rank_out());
        assert_eq!(res.dnum(), a.dnum());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(tsk.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());

        for row in 0..res.dnum().into() {
            self.glwe_copy(&mut res.at_mut(row, 0), &a.at(row, 0));
        }

        self.ggsw_expand_row(res, tsk, scratch);
    }
}

pub trait GGSWFromGGLWE<BE: Backend> {
    fn ggsw_from_gglwe_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_from_gglwe<R, A, T>(&self, res: &mut R, a: &A, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGLWEToRef,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
}

pub trait GGSWExpandRows<BE: Backend> {
    fn ggsw_expand_rows_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row<R, T>(&self, res: &mut R, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> GGSWExpandRows<BE> for Module<BE>
where
    Self: GGLWEProduct<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigBytesOf
        + VecZnxDftBytesOf
        + VecZnxDftApply<BE>
        + VecZnxNormalize<BE>
        + VecZnxBigAddSmallInplace<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxCopy,
{
    fn ggsw_expand_rows_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        let base2k_tsk: usize = tsk_infos.base2k().into();

        let rank: usize = res_infos.rank().into();
        let cols: usize = rank + 1;

        let res_size: usize = res_infos.size();
        let a_size: usize = res_infos.max_k().as_usize().div_ceil(base2k_tsk);

        let a_0: usize = VecZnx::bytes_of(self.n(), 1, a_size);
        let a_dft: usize = self.bytes_of_vec_znx_dft(cols - 1, a_size);
        let res_dft: usize = self.bytes_of_vec_znx_dft(cols, a_size);
        let gglwe_prod: usize = self.gglwe_product_dft_tmp_bytes(res_size, a_size, tsk_infos);
        let normalize: usize = self.vec_znx_big_normalize_tmp_bytes();

        (a_0 + a_dft + res_dft + gglwe_prod).max(normalize)
    }

    fn ggsw_expand_row<R, T>(&self, res: &mut R, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let tsk: &GGLWEToGGSWKeyPrepared<&[u8], BE> = &tsk.to_ref();

        let base2k_res: usize = res.base2k().into();
        let base2k_tsk: usize = tsk.base2k().into();

        assert!(scratch.available() >= self.ggsw_expand_rows_tmp_bytes(res, tsk));

        let rank: usize = res.rank().into();
        let cols: usize = rank + 1;

        let res_conv_size: usize = res.max_k().as_usize().div_ceil(base2k_tsk);

        let (mut a_dft, scratch_1) = scratch.take_vec_znx_dft(self, cols - 1, res_conv_size);
        let (mut a_0, scratch_2) = scratch_1.take_vec_znx(self.n(), 1, res_conv_size);

        // Keyswitch the j-th row of the col 0
        for row in 0..res.dnum().as_usize() {
            let glwe_mi_1: &GLWE<&[u8]> = &res.at(row, 0);

            if base2k_res == base2k_tsk {
                for col_i in 0..cols - 1 {
                    self.vec_znx_dft_apply(1, 0, &mut a_dft, col_i, glwe_mi_1.data(), col_i + 1);
                }
                self.vec_znx_copy(&mut a_0, 0, glwe_mi_1.data(), 0);
            } else {
                for i in 0..cols - 1 {
                    self.vec_znx_normalize(base2k_tsk, &mut a_0, 0, base2k_res, glwe_mi_1.data(), i + 1, scratch_2);
                    self.vec_znx_dft_apply(1, 0, &mut a_dft, i, &a_0, 0);
                }
                self.vec_znx_normalize(base2k_tsk, &mut a_0, 0, base2k_res, glwe_mi_1.data(), 0, scratch_2);
            }

            ggsw_expand_rows_internal(self, row, res, &a_0, &a_dft, tsk, scratch_2)
        }
    }
}

fn ggsw_expand_rows_internal<M, R, C, A, T, BE: Backend>(
    module: &M,
    row: usize,
    res: &mut R,
    a_0: &C,
    a_dft: &A,
    tsk: &T,
    scratch: &mut Scratch<BE>,
) where
    R: GGSWToMut,
    C: VecZnxToRef,
    A: VecZnxDftToRef<BE>,
    M: GGLWEProduct<BE> + VecZnxIdftApplyConsume<BE> + VecZnxBigAddSmallInplace<BE> + VecZnxBigNormalize<BE>,
    T: GGLWEToGGSWKeyPreparedToRef<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
    let a_0: &VecZnx<&[u8]> = &a_0.to_ref();
    let a_dft: &VecZnxDft<&[u8], BE> = &a_dft.to_ref();
    let tsk: &GGLWEToGGSWKeyPrepared<&[u8], BE> = &tsk.to_ref();
    let cols: usize = res.rank().as_usize() + 1;

    // Example for rank 3:
    //
    // Note: M is a vector (m, Bm, B^2m, B^3m, ...), so each column is
    // actually composed of that many dnum and we focus on a specific row here
    // implicitely given ci_dft.
    //
    // # Input
    //
    // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0    , a1    , a2    )
    // col 1: (0, 0, 0, 0)
    // col 2: (0, 0, 0, 0)
    // col 3: (0, 0, 0, 0)
    //
    // # Output
    //
    // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0       , a1       , a2       )
    // col 1: (-(b0s0 + b1s1 + b2s2)       , b0 + M[i], b1       , b2       )
    // col 2: (-(c0s0 + c1s1 + c2s2)       , c0       , c1 + M[i], c2       )
    // col 3: (-(d0s0 + d1s1 + d2s2)       , d0       , d1       , d2 + M[i])
    for col in 1..cols {
        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(module, cols, tsk.size()); // Todo optimise

        // Performs a key-switch for each combination of s[i]*s[j], i.e. for a0, a1, a2
        //
        // # Example for col=1
        //
        // a0 * (-(f0s0 + f1s1 + f1s2) + s0^2, f0, f1, f2) = (-(a0f0s0 + a0f1s1 + a0f1s2) + a0s0^2, a0f0, a0f1, a0f2)
        // +
        // a1 * (-(g0s0 + g1s1 + g1s2) + s0s1, g0, g1, g2) = (-(a1g0s0 + a1g1s1 + a1g1s2) + a1s0s1, a1g0, a1g1, a1g2)
        // +
        // a2 * (-(h0s0 + h1s1 + h1s2) + s0s2, h0, h1, h2) = (-(a2h0s0 + a2h1s1 + a2h1s2) + a2s0s2, a2h0, a2h1, a2h2)
        // =
        // (-(x0s0 + x1s1 + x2s2) + s0(a0s0 + a1s1 + a2s2), x0, x1, x2)
        module.gglwe_product_dft(&mut res_dft, a_dft, tsk.at(col - 1), scratch_1);

        let mut res_big: VecZnxBig<&mut [u8], BE> = module.vec_znx_idft_apply_consume(res_dft);

        // Adds -(sum a[i] * s[i]) + m)  on the i-th column of tmp_idft_i
        //
        // (-(x0s0 + x1s1 + x2s2) + a0s0s0 + a1s0s1 + a2s0s2, x0, x1, x2)
        // +
        // (0, -(a0s0 + a1s1 + a2s2) + M[i], 0, 0)
        // =
        // (-(x0s0 + x1s1 + x2s2) + s0(a0s0 + a1s1 + a2s2), x0 -(a0s0 + a1s1 + a2s2) + M[i], x1, x2)
        // =
        // (-(x0s0 + x1s1 + x2s2), x0 + M[i], x1, x2)
        module.vec_znx_big_add_small_inplace(&mut res_big, col, a_0, 0);

        for j in 0..cols {
            module.vec_znx_big_normalize(
                res.base2k().as_usize(),
                res.at_mut(row, col).data_mut(),
                j,
                tsk.base2k().as_usize(),
                &res_big,
                j,
                scratch_1,
            );
        }
    }
}
