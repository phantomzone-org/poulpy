use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxZero},
};

use crate::layouts::{GGLWEInfos, GGSWInfos, GLWE, GLWESwitchingKey, prepared::GGSWPrepared};

impl GLWESwitchingKey<Vec<u8>> {
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
        GLWE::external_product_scratch_space(
            module,
            &out_infos.glwe_layout(),
            &in_infos.glwe_layout(),
            ggsw_infos,
        )
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
        GLWE::external_product_inplace_scratch_space(module, &out_infos.glwe_layout(), ggsw_infos)
    }
}

impl<DataSelf: DataMut> GLWESwitchingKey<DataSelf> {
    pub fn external_product<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWESwitchingKey<DataLhs>,
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
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            use crate::layouts::GLWEInfos;

            assert_eq!(
                self.rank_in(),
                lhs.rank_in(),
                "ksk_out input rank: {} != ksk_in input rank: {}",
                self.rank_in(),
                lhs.rank_in()
            );
            assert_eq!(
                lhs.rank_out(),
                rhs.rank(),
                "ksk_in output rank: {} != ggsw rank: {}",
                self.rank_out(),
                rhs.rank()
            );
            assert_eq!(
                self.rank_out(),
                rhs.rank(),
                "ksk_out output rank: {} != ggsw rank: {}",
                self.rank_out(),
                rhs.rank()
            );
        }

        (0..self.rank_in().into()).for_each(|col_i| {
            (0..self.dnum().into()).for_each(|row_j| {
                self.at_mut(row_j, col_i)
                    .external_product(module, &lhs.at(row_j, col_i), rhs, scratch);
            });
        });

        (self.dnum().min(lhs.dnum()).into()..self.dnum().into()).for_each(|row_i| {
            (0..self.rank_in().into()).for_each(|col_j| {
                self.at_mut(row_i, col_j).data.zero();
            });
        });
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
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            use crate::layouts::GLWEInfos;

            assert_eq!(
                self.rank_out(),
                rhs.rank(),
                "ksk_out output rank: {} != ggsw rank: {}",
                self.rank_out(),
                rhs.rank()
            );
        }

        (0..self.rank_in().into()).for_each(|col_i| {
            (0..self.dnum().into()).for_each(|row_j| {
                self.at_mut(row_j, col_i)
                    .external_product_inplace(module, rhs, scratch);
            });
        });
    }
}
