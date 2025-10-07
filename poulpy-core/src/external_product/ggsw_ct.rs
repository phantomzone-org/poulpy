use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply,
        VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxZero},
};

use crate::layouts::{GGSWCiphertext, GGSWInfos, GLWECiphertext, GLWEInfos, prepared::GGSWCiphertextPrepared};

impl GGSWCiphertext<Vec<u8>> {
    #[allow(clippy::too_many_arguments)]
    pub fn external_product_scratch_space<B: Backend, OUT, IN, GGSW>(
        module: &Module<B>,
        out_infos: &OUT,
        in_infos: &IN,
        apply_infos: &GGSW,
    ) -> usize
    where
        OUT: GGSWInfos,
        IN: GGSWInfos,
        GGSW: GGSWInfos,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxNormalizeTmpBytes,
    {
        GLWECiphertext::external_product_scratch_space(
            module,
            &out_infos.glwe_layout(),
            &in_infos.glwe_layout(),
            apply_infos,
        )
    }

    pub fn external_product_inplace_scratch_space<B: Backend, OUT, GGSW>(
        module: &Module<B>,
        out_infos: &OUT,
        apply_infos: &GGSW,
    ) -> usize
    where
        OUT: GGSWInfos,
        GGSW: GGSWInfos,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxNormalizeTmpBytes,
    {
        GLWECiphertext::external_product_inplace_scratch_space(module, &out_infos.glwe_layout(), apply_infos)
    }
}

impl<DataSelf: DataMut> GGSWCiphertext<DataSelf> {
    pub fn external_product<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGSWCiphertext<DataLhs>,
        rhs: &GGSWCiphertextPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftApply<B>
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigNormalize<B>
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            use crate::layouts::LWEInfos;

            assert_eq!(lhs.n(), self.n());
            assert_eq!(rhs.n(), self.n());

            assert_eq!(
                self.rank(),
                lhs.rank(),
                "ggsw_out rank: {} != ggsw_in rank: {}",
                self.rank(),
                lhs.rank()
            );
            assert_eq!(
                self.rank(),
                rhs.rank(),
                "ggsw_in rank: {} != ggsw_apply rank: {}",
                self.rank(),
                rhs.rank()
            );

            assert!(scratch.available() >= GGSWCiphertext::external_product_scratch_space(module, self, lhs, rhs))
        }

        let min_dnum: usize = self.dnum().min(lhs.dnum()).into();

        (0..(self.rank() + 1).into()).for_each(|col_i| {
            (0..min_dnum).for_each(|row_j| {
                self.at_mut(row_j, col_i)
                    .external_product(module, &lhs.at(row_j, col_i), rhs, scratch);
            });
            (min_dnum..self.dnum().into()).for_each(|row_i| {
                self.at_mut(row_i, col_i).data.zero();
            });
        });
    }

    pub fn external_product_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGSWCiphertextPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftApply<B>
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigNormalize<B>
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            use crate::layouts::LWEInfos;

            assert_eq!(rhs.n(), self.n());
            assert_eq!(
                self.rank(),
                rhs.rank(),
                "ggsw_out rank: {} != ggsw_apply: {}",
                self.rank(),
                rhs.rank()
            );
        }

        (0..(self.rank() + 1).into()).for_each(|col_i| {
            (0..self.dnum().into()).for_each(|row_j| {
                self.at_mut(row_j, col_i)
                    .external_product_inplace(module, rhs, scratch);
            });
        });
    }
}
