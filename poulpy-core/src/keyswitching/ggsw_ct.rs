use poulpy_hal::{
    api::{
        ScratchAvailable, VecZnxBigAddSmallInplace, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy,
        VecZnxDftAddInplace, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA,
        VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, VecZnx, VmpPMat, ZnxInfos},
};

use crate::{
    layouts::{
        GGLWE, GGLWEInfos, GGSW, GGSWInfos, GLWE, GLWEInfos, LWEInfos,
        prepared::{GLWESwitchingKeyPrepared, TensorKeyPrepared},
    },
    operations::GLWEOperations,
};

impl GGSW<Vec<u8>> {
    pub(crate) fn expand_row_scratch_space<B: Backend, OUT, TSK>(module: &Module<B>, out_infos: &OUT, tsk_infos: &TSK) -> usize
    where
        OUT: GGSWInfos,
        TSK: GGLWEInfos,
        Module<B>: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxBigBytesOf + VecZnxNormalizeTmpBytes,
    {
        let tsk_size: usize = tsk_infos.k().div_ceil(tsk_infos.base2k()) as usize;
        let size_in: usize = out_infos
            .k()
            .div_ceil(tsk_infos.base2k())
            .div_ceil(tsk_infos.dsize().into()) as usize;

        let tmp_dft_i: usize = module.bytes_of_vec_znx_dft((tsk_infos.rank_out() + 1).into(), tsk_size);
        let tmp_a: usize = module.bytes_of_vec_znx_dft(1, size_in);
        let vmp: usize = module.vmp_apply_dft_to_dft_tmp_bytes(
            tsk_size,
            size_in,
            size_in,
            (tsk_infos.rank_in()).into(),  // Verify if rank+1
            (tsk_infos.rank_out()).into(), // Verify if rank+1
            tsk_size,
        );
        let tmp_idft: usize = module.bytes_of_vec_znx_big(1, tsk_size);
        let norm: usize = module.vec_znx_normalize_tmp_bytes();

        tmp_dft_i + ((tmp_a + vmp) | (tmp_idft + norm))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn keyswitch_scratch_space<B: Backend, OUT, IN, KEY, TSK>(
        module: &Module<B>,
        out_infos: &OUT,
        in_infos: &IN,
        apply_infos: &KEY,
        tsk_infos: &TSK,
    ) -> usize
    where
        OUT: GGSWInfos,
        IN: GGSWInfos,
        KEY: GGLWEInfos,
        TSK: GGLWEInfos,
        Module<B>:
            VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxBigBytesOf + VecZnxNormalizeTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(apply_infos.rank_in(), apply_infos.rank_out());
            assert_eq!(tsk_infos.rank_in(), tsk_infos.rank_out());
            assert_eq!(apply_infos.rank_in(), tsk_infos.rank_in());
        }

        let rank: usize = apply_infos.rank_out().into();

        let size_out: usize = out_infos.k().div_ceil(out_infos.base2k()) as usize;
        let res_znx: usize = VecZnx::bytes_of(module.n(), rank + 1, size_out);
        let ci_dft: usize = module.bytes_of_vec_znx_dft(rank + 1, size_out);
        let ks: usize = GLWE::keyswitch_scratch_space(module, out_infos, in_infos, apply_infos);
        let expand_rows: usize = GGSW::expand_row_scratch_space(module, out_infos, tsk_infos);
        let res_dft: usize = module.bytes_of_vec_znx_dft(rank + 1, size_out);

        if in_infos.base2k() == tsk_infos.base2k() {
            res_znx + ci_dft + (ks | expand_rows | res_dft)
        } else {
            let a_conv: usize = VecZnx::bytes_of(
                module.n(),
                1,
                out_infos.k().div_ceil(tsk_infos.base2k()) as usize,
            ) + module.vec_znx_normalize_tmp_bytes();
            res_znx + ci_dft + (a_conv | ks | expand_rows | res_dft)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn keyswitch_inplace_scratch_space<B: Backend, OUT, KEY, TSK>(
        module: &Module<B>,
        out_infos: &OUT,
        apply_infos: &KEY,
        tsk_infos: &TSK,
    ) -> usize
    where
        OUT: GGSWInfos,
        KEY: GGLWEInfos,
        TSK: GGLWEInfos,
        Module<B>:
            VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxBigBytesOf + VecZnxNormalizeTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        GGSW::keyswitch_scratch_space(module, out_infos, out_infos, apply_infos, tsk_infos)
    }
}

impl<DataSelf: DataMut> GGSW<DataSelf> {
    pub fn from_gglwe<DataA, DataTsk, B: Backend>(
        &mut self,
        module: &Module<B>,
        a: &GGLWE<DataA>,
        tsk: &TensorKeyPrepared<DataTsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        DataA: DataRef,
        DataTsk: DataRef,
        Module<B>: VecZnxCopy
            + VecZnxDftBytesOf
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigBytesOf
            + VecZnxNormalizeTmpBytes
            + VecZnxDftApply<B>
            + VecZnxDftCopy<B>
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftAddInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxIdftApplyTmpA<B>
            + VecZnxNormalize<B>,
        Scratch<B>: ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            use crate::layouts::{GLWEInfos, LWEInfos};

            assert_eq!(self.rank(), a.rank_out());
            assert_eq!(self.dnum(), a.dnum());
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(a.n(), module.n() as u32);
            assert_eq!(tsk.n(), module.n() as u32);
        }
        (0..self.dnum().into()).for_each(|row_i| {
            self.at_mut(row_i, 0).copy(module, &a.at(row_i, 0));
        });
        self.expand_row(module, tsk, scratch);
    }

    pub fn keyswitch<DataLhs: DataRef, DataKsk: DataRef, DataTsk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGSW<DataLhs>,
        ksk: &GLWESwitchingKeyPrepared<DataKsk, B>,
        tsk: &TensorKeyPrepared<DataTsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftBytesOf
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxDftBytesOf
            + VecZnxBigBytesOf
            + VecZnxNormalizeTmpBytes
            + VecZnxDftCopy<B>
            + VecZnxDftAddInplace<B>
            + VecZnxIdftApplyTmpA<B>
            + VecZnxNormalize<B>,
        Scratch<B>: ScratchAvailable,
    {
        (0..lhs.dnum().into()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0s0' + a1s1' + a2s2') + M[i], a0, a1, a2)
            self.at_mut(row_i, 0)
                .keyswitch(module, &lhs.at(row_i, 0), ksk, scratch);
        });
        self.expand_row(module, tsk, scratch);
    }

    pub fn keyswitch_inplace<DataKsk: DataRef, DataTsk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        ksk: &GLWESwitchingKeyPrepared<DataKsk, B>,
        tsk: &TensorKeyPrepared<DataTsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftBytesOf
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxDftBytesOf
            + VecZnxBigBytesOf
            + VecZnxNormalizeTmpBytes
            + VecZnxDftCopy<B>
            + VecZnxDftAddInplace<B>
            + VecZnxIdftApplyTmpA<B>
            + VecZnxNormalize<B>,
        Scratch<B>: ScratchAvailable,
    {
        (0..self.dnum().into()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0s0' + a1s1' + a2s2') + M[i], a0, a1, a2)
            self.at_mut(row_i, 0)
                .keyswitch_inplace(module, ksk, scratch);
        });
        self.expand_row(module, tsk, scratch);
    }

    pub fn expand_row<DataTsk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        tsk: &TensorKeyPrepared<DataTsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftBytesOf
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigBytesOf
            + VecZnxNormalizeTmpBytes
            + VecZnxDftApply<B>
            + VecZnxDftCopy<B>
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftAddInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxIdftApplyTmpA<B>
            + VecZnxNormalize<B>,
        Scratch<B>: ScratchAvailable,
    {
        let basek_in: usize = self.base2k().into();
        let basek_tsk: usize = tsk.base2k().into();

        assert!(scratch.available() >= GGSW::expand_row_scratch_space(module, self, tsk));

        let n: usize = self.n().into();
        let rank: usize = self.rank().into();
        let cols: usize = rank + 1;

        let a_size: usize = (self.size() * basek_in).div_ceil(basek_tsk);

        // Keyswitch the j-th row of the col 0
        for row_i in 0..self.dnum().into() {
            let a = &self.at(row_i, 0).data;

            // Pre-compute DFT of (a0, a1, a2)
            let (mut ci_dft, scratch_1) = scratch.take_vec_znx_dft(n, cols, a_size);

            if basek_in == basek_tsk {
                for i in 0..cols {
                    module.vec_znx_dft_apply(1, 0, &mut ci_dft, i, a, i);
                }
            } else {
                let (mut a_conv, scratch_2) = scratch_1.take_vec_znx(n, 1, a_size);
                for i in 0..cols {
                    module.vec_znx_normalize(basek_tsk, &mut a_conv, 0, basek_in, a, i, scratch_2);
                    module.vec_znx_dft_apply(1, 0, &mut ci_dft, i, &a_conv, 0);
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

                let (mut tmp_dft_i, scratch_2) = scratch_1.take_vec_znx_dft(n, cols, tsk.size());
                let (mut tmp_a, scratch_3) = scratch_2.take_vec_znx_dft(n, 1, ci_dft.size().div_ceil(dsize));

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
                        let pmat: &VmpPMat<DataTsk, B> = &tsk.at(col_i - 1, col_j - 1).key.data; // Selects Enc(s[i]s[j])

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

                            module.vec_znx_dft_copy(dsize, dsize - 1 - di, &mut tmp_a, 0, &ci_dft, col_i);
                            if di == 0 && col_i == 1 {
                                module.vmp_apply_dft_to_dft(&mut tmp_dft_i, &tmp_a, pmat, scratch_3);
                            } else {
                                module.vmp_apply_dft_to_dft_add(&mut tmp_dft_i, &tmp_a, pmat, di, scratch_3);
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
                module.vec_znx_dft_add_inplace(&mut tmp_dft_i, col_j, &ci_dft, 0);
                let (mut tmp_idft, scratch_3) = scratch_2.take_vec_znx_big(n, 1, tsk.size());
                for i in 0..cols {
                    module.vec_znx_idft_apply_tmpa(&mut tmp_idft, 0, &mut tmp_dft_i, i);
                    module.vec_znx_big_normalize(
                        basek_in,
                        &mut self.at_mut(row_i, col_j).data,
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
