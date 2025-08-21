use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnxDft, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftFromVecZnx,
        VecZnxDftToVecZnxBigConsume, VecZnxNormalizeTmpBytes, VmpApply, VmpApplyAdd, VmpApplyTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{GGLWEAutomorphismKey, GGLWESwitchingKey, prepared::GGSWCiphertextPrepared};

impl GGLWEAutomorphismKey<Vec<u8>> {
    #[allow(clippy::too_many_arguments)]
    pub fn external_product_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        ggsw_k: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyTmpBytes + VecZnxNormalizeTmpBytes,
    {
        GGLWESwitchingKey::external_product_scratch_space(module, basek, k_out, k_in, ggsw_k, digits, rank)
    }

    pub fn external_product_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        ggsw_k: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyTmpBytes + VecZnxNormalizeTmpBytes,
    {
        GGLWESwitchingKey::external_product_inplace_scratch_space(module, basek, k_out, ggsw_k, digits, rank)
    }
}

impl<DataSelf: DataMut> GGLWEAutomorphismKey<DataSelf> {
    pub fn external_product<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGLWEAutomorphismKey<DataLhs>,
        rhs: &GGSWCiphertextPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftFromVecZnx<B>
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        self.key.external_product(module, &lhs.key, rhs, scratch);
    }

    pub fn external_product_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGSWCiphertextPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftFromVecZnx<B>
            + VmpApply<B>
            + VmpApplyAdd<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        self.key.external_product_inplace(module, rhs, scratch);
    }
}
