use backend::{
    Backend, Module, Scratch, ScratchAvailable, ScratchTakeVecZnxBig, ScratchTakeVecZnxDft, VecZnxAllocBytes,
    VecZnxBigAllocBytes, VecZnxDft, VecZnxDftAddInplace, VecZnxDftCopy, VecZnxDftToVecZnxBigTmpA, VecZnxNormalizeTmpBytes,
    VmpPMat, ZnxInfos,
};

use crate::{GGSWCiphertext, GLWECiphertext, GLWEKeyswitchFamily, GLWESwitchingKeyExec, GLWETensorKeyExec, Infos};

pub trait GGSWKeySwitchFamily<B> =
    GLWEKeyswitchFamily<B> + VecZnxBigAllocBytes + VecZnxDftCopy<B> + VecZnxDftAddInplace<B> + VecZnxDftToVecZnxBigTmpA<B>;

impl GGSWCiphertext<Vec<u8>> {
    pub(crate) fn expand_row_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        self_k: usize,
        k_tsk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GGSWKeySwitchFamily<B>,
    {
        let tsk_size: usize = k_tsk.div_ceil(basek);
        let self_size_out: usize = self_k.div_ceil(basek);
        let self_size_in: usize = self_size_out.div_ceil(digits);

        let tmp_dft_i: usize = module.vec_znx_dft_alloc_bytes(rank + 1, tsk_size);
        let tmp_a: usize = module.vec_znx_dft_alloc_bytes(1, self_size_in);
        let vmp: usize = module.vmp_apply_tmp_bytes(
            self_size_out,
            self_size_in,
            self_size_in,
            rank,
            rank,
            tsk_size,
        );
        let tmp_idft: usize = module.vec_znx_big_alloc_bytes(1, tsk_size);
        let norm: usize = module.vec_znx_normalize_tmp_bytes();
        tmp_dft_i + ((tmp_a + vmp) | (tmp_idft + norm))
    }

    pub fn keyswitch_scratch_space<B: Backend>(
        module: &Module<B>,
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
        Module<B>: GLWEKeyswitchFamily<B> + GGSWKeySwitchFamily<B>,
    {
        let out_size: usize = k_out.div_ceil(basek);
        let res_znx: usize = module.vec_znx_alloc_bytes(rank + 1, out_size);
        let ci_dft: usize = module.vec_znx_dft_alloc_bytes(rank + 1, out_size);
        let ks: usize = GLWECiphertext::keyswitch_scratch_space(module, basek, k_out, k_in, k_ksk, digits_ksk, rank, rank);
        let expand_rows: usize = GGSWCiphertext::expand_row_scratch_space(module, basek, k_out, k_tsk, digits_tsk, rank);
        let res_dft: usize = module.vec_znx_dft_alloc_bytes(rank + 1, out_size);
        res_znx + ci_dft + (ks | expand_rows | res_dft)
    }

    pub fn keyswitch_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits_ksk: usize,
        k_tsk: usize,
        digits_tsk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B> + GGSWKeySwitchFamily<B>,
    {
        GGSWCiphertext::keyswitch_scratch_space(
            module, basek, k_out, k_out, k_ksk, digits_ksk, k_tsk, digits_tsk, rank,
        )
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GGSWCiphertext<DataSelf> {
    pub(crate) fn expand_row<DataCi: AsRef<[u8]>, DataTsk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        row_i: usize,
        col_j: usize,
        ci_dft: &VecZnxDft<DataCi, B>,
        tsk: &GLWETensorKeyExec<DataTsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GGSWKeySwitchFamily<B>,
        Scratch<B>: ScratchTakeVecZnxDft<B> + ScratchTakeVecZnxBig<B>,
    {
        let cols: usize = self.rank() + 1;

        assert!(
            scratch.available()
                >= GGSWCiphertext::expand_row_scratch_space(
                    module,
                    self.basek(),
                    self.k(),
                    tsk.k(),
                    tsk.digits(),
                    tsk.rank()
                )
        );

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

        let (mut tmp_dft_i, scratch1) = scratch.take_vec_znx_dft(module, cols, tsk.size());
        let (mut tmp_a, scratch2) = scratch1.take_vec_znx_dft(module, 1, (ci_dft.size() + digits - 1) / digits);

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

                    module.vec_znx_dft_copy(digits, digits - 1 - di, &mut tmp_a, 0, ci_dft, col_i);
                    if di == 0 && col_i == 1 {
                        module.vmp_apply(&mut tmp_dft_i, &tmp_a, pmat, scratch2);
                    } else {
                        module.vmp_apply_add(&mut tmp_dft_i, &tmp_a, pmat, di, scratch2);
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
        module.vec_znx_dft_add_inplace(&mut tmp_dft_i, col_j, ci_dft, 0);
        let (mut tmp_idft, scratch2) = scratch1.take_vec_znx_big(module, 1, tsk.size());
        (0..cols).for_each(|i| {
            module.vec_znx_dft_to_vec_znx_big_tmp_a(&mut tmp_idft, 0, &mut tmp_dft_i, i);
            module.vec_znx_big_normalize(
                self.basek(),
                &mut self.at_mut(row_i, col_j).data,
                i,
                &tmp_idft,
                0,
                scratch2,
            );
        });
    }

    pub fn keyswitch<DataLhs: AsRef<[u8]>, DataKsk: AsRef<[u8]>, DataTsk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGSWCiphertext<DataLhs>,
        ksk: &GLWESwitchingKeyExec<DataKsk, B>,
        tsk: &GLWETensorKeyExec<DataTsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B> + GGSWKeySwitchFamily<B>,
        Scratch<B>: ScratchTakeVecZnxDft<B> + ScratchTakeVecZnxBig<B>,
    {
        let rank: usize = self.rank();
        let cols: usize = rank + 1;

        // Keyswitch the j-th row of the col 0
        (0..lhs.rows()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0s0' + a1s1' + a2s2') + M[i], a0, a1, a2)
            self.at_mut(row_i, 0)
                .keyswitch(module, &lhs.at(row_i, 0), ksk, scratch);

            // Pre-compute DFT of (a0, a1, a2)
            let (mut ci_dft, scratch1) = scratch.take_vec_znx_dft(module, cols, self.size());
            (0..cols).for_each(|i| {
                module.vec_znx_dft_from_vec_znx(1, 0, &mut ci_dft, i, &self.at(row_i, 0).data, i);
            });
            // Generates
            //
            // col 1: (-(b0s0' + b1s1' + b2s2')    , b0 + M[i], b1       , b2       )
            // col 2: (-(c0s0' + c1s1' + c2s2')    , c0       , c1 + M[i], c2       )
            // col 3: (-(d0s0' + d1s1' + d2s2')    , d0       , d1       , d2 + M[i])
            (1..cols).for_each(|col_j| {
                self.expand_row(module, row_i, col_j, &ci_dft, tsk, scratch1);
            });
        })
    }

    pub fn keyswitch_inplace<DataKsk: AsRef<[u8]>, DataTsk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        ksk: &GLWESwitchingKeyExec<DataKsk, B>,
        tsk: &GLWETensorKeyExec<DataTsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B> + GGSWKeySwitchFamily<B>,
        Scratch<B>: ScratchTakeVecZnxDft<B> + ScratchTakeVecZnxBig<B>,
    {
        unsafe {
            let self_ptr: *mut GGSWCiphertext<DataSelf> = self as *mut GGSWCiphertext<DataSelf>;
            self.keyswitch(module, &*self_ptr, ksk, tsk, scratch);
        }
    }
}
