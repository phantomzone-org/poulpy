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
    prepared::{GGLWEAutomorphismKeyPrepared, GGLWETensorKeyPrepared}, GGLWEMetadata, GGSWCiphertext, GGSWMetadata, GLWECiphertext, Infos
};

impl GGSWCiphertext<Vec<u8>> {
    #[allow(clippy::too_many_arguments)]
    pub fn automorphism_scratch_space<B: Backend>(
        module: &Module<B>,
        out_metadata: GGSWMetadata,
        in_metadata: GGSWMetadata,
        key_metadata: GGLWEMetadata,
        tsk_metadata: GGLWEMetadata,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxBigNormalizeTmpBytes,
    {
        let out_size: usize = out_metadata.k.div_ceil(out_metadata.basek);
        let ci_dft: usize = module.vec_znx_dft_alloc_bytes(key_metadata.rank_out + 1, out_size);
        let ks_internal: usize = GLWECiphertext::keyswitch_scratch_space(module, out_metadata.as_glwe(), in_metadata.as_glwe(), key_metadata);
        let expand: usize = GGSWCiphertext::expand_row_scratch_space(module, out_metadata, tsk_metadata);
        ci_dft + (ks_internal | expand)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn automorphism_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        out_metadata: GGSWMetadata,
        key_metadata: GGLWEMetadata,
        tsk_metadata: GGLWEMetadata,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxBigNormalizeTmpBytes,
    {
        GGSWCiphertext::automorphism_scratch_space(
            module,
            out_metadata,
            out_metadata,
            key_metadata,
            tsk_metadata,
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
                        self.metadata(),
                        lhs.metadata(),
                        auto_key.metadata(),
                        tensor_key.metadata()
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
