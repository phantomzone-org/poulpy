use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, ScratchTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftAddInplace, VecZnxDftApply,
        VecZnxDftBytesOf, VecZnxDftCopy, VecZnxIdftApplyTmpA, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, Module, Scratch, VmpPMat, ZnxInfos},
};

use crate::{
    GLWECopy, ScratchTakeCore,
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToRef, GGSW, GGSWInfos, GGSWToMut, GLWEInfos, LWEInfos,
        prepared::{TensorKeyPrepared, TensorKeyPreparedToRef},
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
        T: TensorKeyPreparedToRef<BE>,
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
        T: TensorKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGLWE<&[u8]> = &a.to_ref();
        let tsk: &TensorKeyPrepared<&[u8], BE> = &tsk.to_ref();

        assert_eq!(res.rank(), a.rank_out());
        assert_eq!(res.dnum(), a.dnum());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(tsk.n(), self.n() as u32);

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
        T: TensorKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> GGSWExpandRows<BE> for Module<BE> where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VecZnxDftCopy<BE>
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VecZnxDftAddInplace<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxNormalize<BE>
{
}

pub trait GGSWExpandRows<BE: Backend>
where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VecZnxDftCopy<BE>
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VecZnxDftAddInplace<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxNormalize<BE>,
{
    fn ggsw_expand_rows_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        let tsk_size: usize = tsk_infos.k().div_ceil(tsk_infos.base2k()) as usize;
        let size_in: usize = res_infos
            .k()
            .div_ceil(tsk_infos.base2k())
            .div_ceil(tsk_infos.dsize().into()) as usize;

        let tmp_dft_i: usize = self.bytes_of_vec_znx_dft((tsk_infos.rank_out() + 1).into(), tsk_size);
        let tmp_a: usize = self.bytes_of_vec_znx_dft(1, size_in);
        let vmp: usize = self.vmp_apply_dft_to_dft_tmp_bytes(
            tsk_size,
            size_in,
            size_in,
            (tsk_infos.rank_in()).into(),  // Verify if rank+1
            (tsk_infos.rank_out()).into(), // Verify if rank+1
            tsk_size,
        );
        let tmp_idft: usize = self.bytes_of_vec_znx_big(1, tsk_size);
        let norm: usize = self.vec_znx_normalize_tmp_bytes();

        tmp_dft_i + ((tmp_a + vmp) | (tmp_idft + norm))
    }

    fn ggsw_expand_row<R, T>(&self, res: &mut R, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        T: TensorKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let tsk: &TensorKeyPrepared<&[u8], BE> = &tsk.to_ref();

        let basek_in: usize = res.base2k().into();
        let basek_tsk: usize = tsk.base2k().into();

        assert!(scratch.available() >= self.ggsw_expand_rows_tmp_bytes(res, tsk));

        let rank: usize = res.rank().into();
        let cols: usize = rank + 1;

        let a_size: usize = (res.size() * basek_in).div_ceil(basek_tsk);

        // Keyswitch the j-th row of the col 0
        for row_i in 0..res.dnum().into() {
            let a = &res.at(row_i, 0).data;

            // Pre-compute DFT of (a0, a1, a2)
            let (mut ci_dft, scratch_1) = scratch.take_vec_znx_dft(self, cols, a_size);

            if basek_in == basek_tsk {
                for i in 0..cols {
                    self.vec_znx_dft_apply(1, 0, &mut ci_dft, i, a, i);
                }
            } else {
                let (mut a_conv, scratch_2) = scratch_1.take_vec_znx(self, 1, a_size);
                for i in 0..cols {
                    self.vec_znx_normalize(basek_tsk, &mut a_conv, 0, basek_in, a, i, scratch_2);
                    self.vec_znx_dft_apply(1, 0, &mut ci_dft, i, &a_conv, 0);
                }
            }

            for col_j in 1..cols {
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

                let dsize: usize = tsk.dsize().into();

                let (mut tmp_dft_i, scratch_2) = scratch_1.take_vec_znx_dft(self, cols, tsk.size());
                let (mut tmp_a, scratch_3) = scratch_2.take_vec_znx_dft(self, 1, ci_dft.size().div_ceil(dsize));

                {
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
                    for col_i in 1..cols {
                        let pmat: &VmpPMat<&[u8], BE> = &tsk.at(col_i - 1, col_j - 1).key.data; // Selects Enc(s[i]s[j])

                        // Extracts a[i] and multipies with Enc(s[i]s[j])
                        for di in 0..dsize {
                            tmp_a.set_size((ci_dft.size() + di) / dsize);

                            // Small optimization for dsize > 2
                            // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                            // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(dsize-1) * B}.
                            // As such we can ignore the last dsize-2 limbs safely of the sum of vmp products.
                            // It is possible to further ignore the last dsize-1 limbs, but this introduce
                            // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                            // noise is kept with respect to the ideal functionality.
                            tmp_dft_i.set_size(tsk.size() - ((dsize - di) as isize - 2).max(0) as usize);

                            self.vec_znx_dft_copy(dsize, dsize - 1 - di, &mut tmp_a, 0, &ci_dft, col_i);
                            if di == 0 && col_i == 1 {
                                self.vmp_apply_dft_to_dft(&mut tmp_dft_i, &tmp_a, pmat, scratch_3);
                            } else {
                                self.vmp_apply_dft_to_dft_add(&mut tmp_dft_i, &tmp_a, pmat, di, scratch_3);
                            }
                        }
                    }
                }

                // Adds -(sum a[i] * s[i]) + m)  on the i-th column of tmp_idft_i
                //
                // (-(x0s0 + x1s1 + x2s2) + a0s0s0 + a1s0s1 + a2s0s2, x0, x1, x2)
                // +
                // (0, -(a0s0 + a1s1 + a2s2) + M[i], 0, 0)
                // =
                // (-(x0s0 + x1s1 + x2s2) + s0(a0s0 + a1s1 + a2s2), x0 -(a0s0 + a1s1 + a2s2) + M[i], x1, x2)
                // =
                // (-(x0s0 + x1s1 + x2s2), x0 + M[i], x1, x2)
                self.vec_znx_dft_add_inplace(&mut tmp_dft_i, col_j, &ci_dft, 0);
                let (mut tmp_idft, scratch_3) = scratch_2.take_vec_znx_big(self, 1, tsk.size());
                for i in 0..cols {
                    self.vec_znx_idft_apply_tmpa(&mut tmp_idft, 0, &mut tmp_dft_i, i);
                    self.vec_znx_big_normalize(
                        basek_in,
                        &mut res.at_mut(row_i, col_j).data,
                        i,
                        basek_tsk,
                        &tmp_idft,
                        0,
                        scratch_3,
                    );
                }
            }
        }
    }
}
