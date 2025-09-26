use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxZero},
};

use crate::layouts::{
    GGLWEAutomorphismKey, GGLWELayoutInfos, GGLWESwitchingKey, GLWECiphertext, GLWEInfos,
    prepared::{GGLWEAutomorphismKeyPrepared, GGLWESwitchingKeyPrepared},
};

impl GGLWEAutomorphismKey<Vec<u8>> {
    pub fn keyswitch_scratch_space<B: Backend, OUT, IN, KEY>(
        module: &Module<B>,
        out_infos: &OUT,
        in_infos: &IN,
        key_infos: &KEY,
    ) -> usize
    where
        OUT: GGLWELayoutInfos,
        IN: GGLWELayoutInfos,
        KEY: GGLWELayoutInfos,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        GGLWESwitchingKey::keyswitch_scratch_space(module, out_infos, in_infos, key_infos)
    }

    pub fn keyswitch_inplace_scratch_space<B: Backend, OUT, KEY>(module: &Module<B>, out_infos: &OUT, key_infos: &KEY) -> usize
    where
        OUT: GGLWELayoutInfos,
        KEY: GGLWELayoutInfos,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        GGLWESwitchingKey::keyswitch_inplace_scratch_space(module, out_infos, key_infos)
    }
}

impl<DataSelf: DataMut> GGLWEAutomorphismKey<DataSelf> {
    pub fn keyswitch<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGLWEAutomorphismKey<DataLhs>,
        rhs: &GGLWESwitchingKeyPrepared<DataRhs, B>,
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
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        self.key.keyswitch(module, &lhs.key, rhs, scratch);
    }

    pub fn keyswitch_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGLWEAutomorphismKeyPrepared<DataRhs, B>,
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
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        self.key.keyswitch_inplace(module, &rhs.key, scratch);
    }
}

impl GGLWESwitchingKey<Vec<u8>> {
    pub fn keyswitch_scratch_space<B: Backend, OUT, IN, KEY>(
        module: &Module<B>,
        out_infos: &OUT,
        in_infos: &IN,
        key_apply: &KEY,
    ) -> usize
    where
        OUT: GGLWELayoutInfos,
        IN: GGLWELayoutInfos,
        KEY: GGLWELayoutInfos,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        GLWECiphertext::keyswitch_scratch_space(module, out_infos, in_infos, key_apply)
    }

    pub fn keyswitch_inplace_scratch_space<B: Backend, OUT, KEY>(module: &Module<B>, out_infos: &OUT, key_apply: &KEY) -> usize
    where
        OUT: GGLWELayoutInfos + GLWEInfos,
        KEY: GGLWELayoutInfos + GLWEInfos,
        Module<B>: VecZnxDftAllocBytes + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        GLWECiphertext::keyswitch_inplace_scratch_space(module, out_infos, key_apply)
    }
}

impl<DataSelf: DataMut> GGLWESwitchingKey<DataSelf> {
    pub fn keyswitch<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGLWESwitchingKey<DataLhs>,
        rhs: &GGLWESwitchingKeyPrepared<DataRhs, B>,
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
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes,
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B> + TakeVecZnx,
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
                rhs.rank_in(),
                "ksk_in output rank: {} != ksk_apply input rank: {}",
                self.rank_out(),
                rhs.rank_in()
            );
            assert_eq!(
                self.rank_out(),
                rhs.rank_out(),
                "ksk_out output rank: {} != ksk_apply output rank: {}",
                self.rank_out(),
                rhs.rank_out()
            );
            assert!(
                self.rows() <= lhs.rows(),
                "self.rows()={} > lhs.rows()={}",
                self.rows(),
                lhs.rows()
            );
            assert_eq!(
                self.digits(),
                lhs.digits(),
                "ksk_out digits: {} != ksk_in digits: {}",
                self.digits(),
                lhs.digits()
            )
        }

        (0..self.rank_in().into()).for_each(|col_i| {
            (0..self.rows().into()).for_each(|row_j| {
                self.at_mut(row_j, col_i)
                    .keyswitch(module, &lhs.at(row_j, col_i), rhs, scratch);
            });
        });

        (self.rows().min(lhs.rows()).into()..self.rows().into()).for_each(|row_i| {
            (0..self.rank_in().into()).for_each(|col_j| {
                self.at_mut(row_i, col_j).data.zero();
            });
        });
    }

    pub fn keyswitch_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGLWESwitchingKeyPrepared<DataRhs, B>,
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
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes,
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B> + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank_out(),
                rhs.rank_out(),
                "ksk_out output rank: {} != ksk_apply output rank: {}",
                self.rank_out(),
                rhs.rank_out()
            );
        }

        (0..self.rank_in().into()).for_each(|col_i| {
            (0..self.rows().into()).for_each(|row_j| {
                self.at_mut(row_j, col_i)
                    .keyswitch_inplace(module, rhs, scratch)
            });
        });
    }
}
