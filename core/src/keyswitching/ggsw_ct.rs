use backend::hal::{
    api::{
        ScratchAvailable, TakeVecZnxBig, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAddInplace, VecZnxDftAllocBytes, VecZnxDftCopy, VecZnxDftFromVecZnx,
        VecZnxDftToVecZnxBigConsume, VecZnxDftToVecZnxBigTmpA, VecZnxNormalizeTmpBytes, VmpApply, VmpApplyAdd, VmpApplyTmpBytes,
        ZnxInfos,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, VecZnx, VmpPMat},
};

use crate::{
    layouts::{
        GGLWECiphertext, GGSWCiphertext, GLWECiphertext, Infos,
        prepared::{GGLWESwitchingKeyPrepared, GGLWETensorKeyPrepared},
    },
    operations::GLWEOperations,
};

impl GGSWCiphertext<Vec<u8>> {
    pub(crate) fn expand_row_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        self_k: usize,
        k_tsk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyTmpBytes + VecZnxBigAllocBytes + VecZnxNormalizeTmpBytes,
    {
        let tsk_size: usize = k_tsk.div_ceil(basek);
        let self_size_out: usize = self_k.div_ceil(basek);
        let self_size_in: usize = self_size_out.div_ceil(digits);

        let tmp_dft_i: usize = module.vec_znx_dft_alloc_bytes(n, rank + 1, tsk_size);
        let tmp_a: usize = module.vec_znx_dft_alloc_bytes(n, 1, self_size_in);
        let vmp: usize = module.vmp_apply_tmp_bytes(
            n,
            self_size_out,
            self_size_in,
            self_size_in,
            rank,
            rank,
            tsk_size,
        );
        let tmp_idft: usize = module.vec_znx_big_alloc_bytes(n, 1, tsk_size);
        let norm: usize = module.vec_znx_normalize_tmp_bytes(n);
        tmp_dft_i + ((tmp_a + vmp) | (tmp_idft + norm))
    }

    pub fn keyswitch_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits_ksk: usize,
        k_tsk: usize,
        digits_tsk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>:
            VecZnxDftAllocBytes + VmpApplyTmpBytes + VecZnxBigAllocBytes + VecZnxNormalizeTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        let out_size: usize = k_out.div_ceil(basek);
        let res_znx: usize = VecZnx::alloc_bytes(n, rank + 1, out_size);
        let ci_dft: usize = module.vec_znx_dft_alloc_bytes(n, rank + 1, out_size);
        let ks: usize = GLWECiphertext::keyswitch_scratch_space(module, n, basek, k_out, k_in, k_ksk, digits_ksk, rank, rank);
        let expand_rows: usize = GGSWCiphertext::expand_row_scratch_space(module, n, basek, k_out, k_tsk, digits_tsk, rank);
        let res_dft: usize = module.vec_znx_dft_alloc_bytes(n, rank + 1, out_size);
        res_znx + ci_dft + (ks | expand_rows | res_dft)
    }

    pub fn keyswitch_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits_ksk: usize,
        k_tsk: usize,
        digits_tsk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>:
            VecZnxDftAllocBytes + VmpApplyTmpBytes + VecZnxBigAllocBytes + VecZnxNormalizeTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        GGSWCiphertext::keyswitch_scratch_space(
            module, n, basek, k_out, k_out, k_ksk, digits_ksk, k_tsk, digits_tsk, rank,
        )
    }
}

impl<DataSelf: DataMut> GGSWCiphertext<DataSelf> {
    pub fn from_gglwe<DataA, DataTsk, B: Backend>(
        &mut self,
        module: &Module<B>,
        a: &GGLWECiphertext<DataA>,
        tsk: &GGLWETensorKeyPrepared<DataTsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        DataA: DataRef,
        DataTsk: DataRef,
        Module<B>: VecZnxCopy
            + VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftCopy<B>
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftAddInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxDftToVecZnxBigTmpA<B>,
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B> + TakeVecZnxBig<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), a.rank());
            assert_eq!(self.rows(), a.rows());
            assert_eq!(self.n(), module.n());
            assert_eq!(a.n(), module.n());
            assert_eq!(tsk.n(), module.n());
        }
        (0..self.rows()).for_each(|row_i| {
            self.at_mut(row_i, 0).copy(module, &a.at(row_i, 0));
        });
        self.expand_row(module, tsk, scratch);
    }

    pub fn keyswitch<DataLhs: DataRef, DataKsk: DataRef, DataTsk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGSWCiphertext<DataLhs>,
        ksk: &GGLWESwitchingKeyPrepared<DataKsk, B>,
        tsk: &GGLWETensorKeyPrepared<DataTsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxDftAllocBytes
            + VecZnxBigAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftCopy<B>
            + VecZnxDftAddInplace<B>
            + VecZnxDftToVecZnxBigTmpA<B>,
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B> + TakeVecZnxBig<B>,
    {
        self.keyswitch_internal(module, lhs, ksk, scratch);
        self.expand_row(module, tsk, scratch);
    }

    pub fn keyswitch_inplace<DataKsk: DataRef, DataTsk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        ksk: &GGLWESwitchingKeyPrepared<DataKsk, B>,
        tsk: &GGLWETensorKeyPrepared<DataTsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxDftAllocBytes
            + VecZnxBigAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftCopy<B>
            + VecZnxDftAddInplace<B>
            + VecZnxDftToVecZnxBigTmpA<B>,
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B> + TakeVecZnxBig<B>,
    {
        unsafe {
            let self_ptr: *mut GGSWCiphertext<DataSelf> = self as *mut GGSWCiphertext<DataSelf>;
            self.keyswitch(module, &*self_ptr, ksk, tsk, scratch);
        }
    }

    pub fn expand_row<DataTsk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        tsk: &GGLWETensorKeyPrepared<DataTsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftCopy<B>
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftAddInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxDftToVecZnxBigTmpA<B>,
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B> + TakeVecZnxBig<B>,
    {
        assert!(
            scratch.available()
                >= GGSWCiphertext::expand_row_scratch_space(
                    module,
                    self.n(),
                    self.basek(),
                    self.k(),
                    tsk.k(),
                    tsk.digits(),
                    tsk.rank()
                )
        );

        let n: usize = self.n();
        let rank: usize = self.rank();
        let cols: usize = rank + 1;

        // Keyswitch the j-th row of the col 0
        (0..self.rows()).for_each(|row_i| {
            // Pre-compute DFT of (a0, a1, a2)
            let (mut ci_dft, scratch1) = scratch.take_vec_znx_dft(n, cols, self.size());
            (0..cols).for_each(|i| {
                module.vec_znx_dft_from_vec_znx(1, 0, &mut ci_dft, i, &self.at(row_i, 0).data, i);
            });

            (1..cols).for_each(|col_j| {
                // Example for rank 3:
                //
                // Note: M is a vector (m, Bm, B^2m, B^3m, ...), so each column is
                // actually composed of that many rows and we focus on a specific row here
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

                let digits: usize = tsk.digits();

                let (mut tmp_dft_i, scratch2) = scratch1.take_vec_znx_dft(n, cols, tsk.size());
                let (mut tmp_a, scratch3) = scratch2.take_vec_znx_dft(n, 1, ci_dft.size().div_ceil(digits));

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
                    (1..cols).for_each(|col_i| {
                        let pmat: &VmpPMat<DataTsk, B> = &tsk.at(col_i - 1, col_j - 1).key.data; // Selects Enc(s[i]s[j])

                        // Extracts a[i] and multipies with Enc(s[i]s[j])
                        (0..digits).for_each(|di| {
                            tmp_a.set_size((ci_dft.size() + di) / digits);

                            // Small optimization for digits > 2
                            // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                            // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(digits-1) * B}.
                            // As such we can ignore the last digits-2 limbs safely of the sum of vmp products.
                            // It is possible to further ignore the last digits-1 limbs, but this introduce
                            // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                            // noise is kept with respect to the ideal functionality.
                            tmp_dft_i.set_size(tsk.size() - ((digits - di) as isize - 2).max(0) as usize);

                            module.vec_znx_dft_copy(digits, digits - 1 - di, &mut tmp_a, 0, &ci_dft, col_i);
                            if di == 0 && col_i == 1 {
                                module.vmp_apply(&mut tmp_dft_i, &tmp_a, pmat, scratch3);
                            } else {
                                module.vmp_apply_add(&mut tmp_dft_i, &tmp_a, pmat, di, scratch3);
                            }
                        });
                    });
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
                let (mut tmp_idft, scratch3) = scratch2.take_vec_znx_big(n, 1, tsk.size());
                (0..cols).for_each(|i| {
                    module.vec_znx_dft_to_vec_znx_big_tmp_a(&mut tmp_idft, 0, &mut tmp_dft_i, i);
                    module.vec_znx_big_normalize(
                        self.basek(),
                        &mut self.at_mut(row_i, col_j).data,
                        i,
                        &tmp_idft,
                        0,
                        scratch3,
                    );
                });
            })
        })
    }

    fn keyswitch_internal<DataLhs: DataRef, DataKsk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGSWCiphertext<DataLhs>,
        ksk: &GGLWESwitchingKeyPrepared<DataKsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>,
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B>,
    {
        // Keyswitch the j-th row of the col 0
        (0..lhs.rows()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0s0' + a1s1' + a2s2') + M[i], a0, a1, a2)
            self.at_mut(row_i, 0)
                .keyswitch(module, &lhs.at(row_i, 0), ksk, scratch);
        })
    }
}
