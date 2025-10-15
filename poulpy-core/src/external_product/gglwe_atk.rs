use poulpy_hal::{
    api::{
        ScratchAvailable, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalize,
        VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{AutomorphismKey, GGLWEInfos, GGSWInfos, GLWESwitchingKey, prepared::GGSWPrepared};

impl AutomorphismKey<Vec<u8>> {
    pub fn external_product_scratch_space<B: Backend, OUT, IN, GGSW>(
        module: &Module<B>,
        out_infos: &OUT,
        in_infos: &IN,
        ggsw_infos: &GGSW,
    ) -> usize
    where
        OUT: GGLWEInfos,
        IN: GGLWEInfos,
        GGSW: GGSWInfos,
        Module<B>: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxNormalizeTmpBytes,
    {
        GLWESwitchingKey::external_product_scratch_space(module, out_infos, in_infos, ggsw_infos)
    }

    pub fn external_product_inplace_scratch_space<B: Backend, OUT, GGSW>(
        module: &Module<B>,
        out_infos: &OUT,
        ggsw_infos: &GGSW,
    ) -> usize
    where
        OUT: GGLWEInfos,
        GGSW: GGSWInfos,
        Module<B>: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxNormalizeTmpBytes,
    {
        GLWESwitchingKey::external_product_inplace_scratch_space(module, out_infos, ggsw_infos)
    }
}

impl<DataSelf: DataMut> AutomorphismKey<DataSelf> {
    pub fn external_product<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &AutomorphismKey<DataLhs>,
        rhs: &GGSWPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftBytesOf
            + VmpApplyDftToDftTmpBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftApply<B>
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigNormalize<B>
            + VecZnxNormalize<B>,
        Scratch<B>: ScratchAvailable,
    {
        self.key.external_product(module, &lhs.key, rhs, scratch);
    }

    pub fn external_product_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGSWPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftBytesOf
            + VmpApplyDftToDftTmpBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftApply<B>
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigNormalize<B>
            + VecZnxNormalize<B>,
        Scratch<B>: ScratchAvailable,
    {
        self.key.external_product_inplace(module, rhs, scratch);
    }
}
