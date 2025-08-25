use poulpy_hal::{
    api::{
        DFT, IDFTConsume, IDFTTmpA, ScratchAvailable, TakeVecZnxBig, TakeVecZnxDft, VecZnxAutomorphismInplace,
        VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddInplace,
        VecZnxDftAllocBytes, VecZnxDftCopy, VecZnxNormalizeTmpBytes, VmpApply, VmpApplyAdd, VmpApplyTmpBytes,
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
        let ci_dft: usize = module.vec_znx_dft_alloc_bytes(rank + 1, out_size);
        let ks_internal: usize =
            GLWECiphertext::keyswitch_scratch_space(module, basek, k_out, k_in, k_ksk, digits_ksk, rank, rank);
        let expand: usize = GGSWCiphertext::expand_row_scratch_space(module, basek, k_out, k_tsk, digits_tsk, rank);
        ci_dft + (ks_internal | expand)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn automorphism_inplace_scratch_space<B: Backend>(
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
        Module<B>:
            VecZnxDftAllocBytes + VmpApplyTmpBytes + VecZnxBigAllocBytes + VecZnxNormalizeTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        GGSWCiphertext::automorphism_scratch_space(
            module, basek, k_out, k_out, k_ksk, digits_ksk, k_tsk, digits_tsk, rank,
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
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + DFT<B>
            + IDFTConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxAutomorphismInplace
            + VecZnxBigAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftCopy<B>
            + VecZnxDftAddInplace<B>
            + IDFTTmpA<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnxBig<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), auto_key.n());
            assert_eq!(lhs.n(), auto_key.n());

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
                        lhs.k(),
                        auto_key.k(),
                        auto_key.digits(),
                        tensor_key.k(),
                        tensor_key.digits(),
                        self.rank(),
                    )
            )
        };

        self.automorphism_internal(module, lhs, auto_key, scratch);
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
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + DFT<B>
            + IDFTConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxAutomorphismInplace
            + VecZnxBigAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftCopy<B>
            + VecZnxDftAddInplace<B>
            + IDFTTmpA<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnxBig<B>,
    {
        unsafe {
            let self_ptr: *mut GGSWCiphertext<DataSelf> = self as *mut GGSWCiphertext<DataSelf>;
            self.automorphism(module, &*self_ptr, auto_key, tensor_key, scratch);
        }
    }

    fn automorphism_internal<DataLhs: DataRef, DataAk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGSWCiphertext<DataLhs>,
        auto_key: &GGLWEAutomorphismKeyPrepared<DataAk, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + DFT<B>
            + IDFTConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxAutomorphismInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        // Keyswitch the j-th row of the col 0
        (0..lhs.rows()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0pi^-1(s0) + a1pi^-1(s1) + a2pi^-1(s2)) + M[i], a0, a1, a2)
            self.at_mut(row_i, 0)
                .automorphism(module, &lhs.at(row_i, 0), auto_key, scratch);
        });
    }
}
