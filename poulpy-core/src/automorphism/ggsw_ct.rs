use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, VecZnxAutomorphismInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddInplace, VecZnxDftAllocBytes,
        VecZnxDftApply, VecZnxDftCopy, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNormalize, VecZnxNormalizeTmpBytes,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    GGSWCiphertext, GLWECiphertext, Infos,
    prepared::{GGLWEAutomorphismKeyPrepared, GGLWETensorKeyPrepared},
};

impl GGSWCiphertext<Vec<u8>> {
    #[allow(clippy::too_many_arguments)]
    pub fn automorphism_scratch_space<B: Backend>(
        module: &Module<B>,
        basek_out: usize,
        k_out: usize,
        basek_in: usize,
        k_in: usize,
        basek_ksk: usize,
        k_ksk: usize,
        digits_ksk: usize,
        basek_tsk: usize,
        k_tsk: usize,
        digits_tsk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxBigNormalizeTmpBytes,
    {
        let out_size: usize = k_out.div_ceil(basek_out);
        let ci_dft: usize = module.vec_znx_dft_alloc_bytes(rank + 1, out_size);
        let ks_internal: usize = GLWECiphertext::keyswitch_scratch_space(
            module, basek_out, k_out, basek_in, k_in, basek_ksk, k_ksk, digits_ksk, rank, rank,
        );
        let expand: usize = GGSWCiphertext::expand_row_scratch_space(module, k_out, basek_tsk, k_tsk, digits_tsk, rank);
        ci_dft + (ks_internal | expand)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn automorphism_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek_out: usize,
        k_out: usize,
        basek_ksk: usize,
        k_ksk: usize,
        digits_ksk: usize,
        basek_tsk: usize,
        k_tsk: usize,
        digits_tsk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxBigNormalizeTmpBytes,
    {
        GGSWCiphertext::automorphism_scratch_space(
            module, basek_out, k_out, basek_out, k_out, basek_ksk, k_ksk, digits_ksk, basek_tsk, k_tsk, digits_tsk, rank,
        )
    }
}

impl<DataSelf: DataMut> GGSWCiphertext<DataSelf> {
    pub fn automorphism<DataLhs: DataRef, DataAk: DataRef, DataTsk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGSWCiphertext<DataLhs>,
        auto_key: &GGLWEAutomorphismKeyPrepared<DataAk, B>,
        tensor_key: &GGLWETensorKeyPrepared<DataTsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxAutomorphismInplace<B>
            + VecZnxBigAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftCopy<B>
            + VecZnxDftAddInplace<B>
            + VecZnxIdftApplyTmpA<B>
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnxBig<B> + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), module.n());
            assert_eq!(lhs.n(), module.n());
            assert_eq!(auto_key.n(), module.n());
            assert_eq!(tensor_key.n(), module.n());

            assert_eq!(
                self.rank(),
                lhs.rank(),
                "ggsw_out rank: {} != ggsw_in rank: {}",
                self.rank(),
                lhs.rank()
            );
            assert_eq!(
                self.rank(),
                auto_key.rank(),
                "ggsw_in rank: {} != auto_key rank: {}",
                self.rank(),
                auto_key.rank()
            );
            assert_eq!(
                self.rank(),
                tensor_key.rank(),
                "ggsw_in rank: {} != tensor_key rank: {}",
                self.rank(),
                tensor_key.rank()
            );
            assert!(
                scratch.available()
                    >= GGSWCiphertext::automorphism_scratch_space(
                        module,
                        self.basek(),
                        self.k(),
                        lhs.basek(),
                        lhs.k(),
                        auto_key.basek(),
                        auto_key.k(),
                        auto_key.digits(),
                        tensor_key.basek(),
                        tensor_key.k(),
                        tensor_key.digits(),
                        self.rank(),
                    )
            )
        };

        // Keyswitch the j-th row of the col 0
        (0..lhs.rows()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0pi^-1(s0) + a1pi^-1(s1) + a2pi^-1(s2)) + M[i], a0, a1, a2)
            self.at_mut(row_i, 0)
                .automorphism(module, &lhs.at(row_i, 0), auto_key, scratch);
        });
        self.expand_row(module, tensor_key, scratch);
    }

    pub fn automorphism_inplace<DataKsk: DataRef, DataTsk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        auto_key: &GGLWEAutomorphismKeyPrepared<DataKsk, B>,
        tensor_key: &GGLWETensorKeyPrepared<DataTsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxAutomorphismInplace<B>
            + VecZnxBigAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftCopy<B>
            + VecZnxDftAddInplace<B>
            + VecZnxIdftApplyTmpA<B>
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnxBig<B> + TakeVecZnx,
    {
        // Keyswitch the j-th row of the col 0
        (0..self.rows()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0pi^-1(s0) + a1pi^-1(s1) + a2pi^-1(s2)) + M[i], a0, a1, a2)
            self.at_mut(row_i, 0)
                .automorphism_inplace(module, auto_key, scratch);
        });
        self.expand_row(module, tensor_key, scratch);
    }
}
