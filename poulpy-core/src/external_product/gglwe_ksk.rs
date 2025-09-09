use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnxDft, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume,
        VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxZero},
};

use crate::layouts::{GGLWESwitchingKey, GLWECiphertext, Infos, prepared::GGSWCiphertextPrepared};

impl GGLWESwitchingKey<Vec<u8>> {
    #[allow(clippy::too_many_arguments)]
    pub fn external_product_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ggsw: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxNormalizeTmpBytes,
    {
        GLWECiphertext::external_product_scratch_space(module, basek, k_out, k_in, k_ggsw, digits, rank)
    }

    pub fn external_product_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_ggsw: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxNormalizeTmpBytes,
    {
        GLWECiphertext::external_product_inplace_scratch_space(module, basek, k_out, k_ggsw, digits, rank)
    }
}

impl<DataSelf: DataMut> GGLWESwitchingKey<DataSelf> {
    pub fn external_product<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGLWESwitchingKey<DataLhs>,
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
            + VecZnxBigNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
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

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                self.at_mut(row_j, col_i)
                    .external_product(module, &lhs.at(row_j, col_i), rhs, scratch);
            });
        });

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank_in()).for_each(|col_j| {
                self.at_mut(row_i, col_j).data.zero();
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
            + VecZnxBigNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank_out(),
                rhs.rank(),
                "ksk_out output rank: {} != ggsw rank: {}",
                self.rank_out(),
                rhs.rank()
            );
        }

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                self.at_mut(row_j, col_i)
                    .external_product_inplace(module, rhs, scratch);
            });
        });
    }
}
