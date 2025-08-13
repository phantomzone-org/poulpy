use backend::hal::{
    api::{ScratchAvailable, TakeVecZnxDft, ZnxZero},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    AutomorphismKey, AutomorphismKeyExec, GLWECiphertext, GLWEKeyswitchFamily, GLWESwitchingKey, GLWESwitchingKeyExec, Infos,
};

impl AutomorphismKey<Vec<u8>> {
    pub fn keyswitch_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        GLWESwitchingKey::keyswitch_scratch_space(module, n, basek, k_out, k_in, k_ksk, digits, rank, rank)
    }

    pub fn keyswitch_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        GLWESwitchingKey::keyswitch_inplace_scratch_space(module, n, basek, k_out, k_ksk, digits, rank)
    }
}

impl<DataSelf: DataMut> AutomorphismKey<DataSelf> {
    pub fn keyswitch<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &AutomorphismKey<DataLhs>,
        rhs: &GLWESwitchingKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        self.key.keyswitch(module, &lhs.key, rhs, scratch);
    }

    pub fn keyswitch_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &AutomorphismKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        self.key.keyswitch_inplace(module, &rhs.key, scratch);
    }
}

impl GLWESwitchingKey<Vec<u8>> {
    pub fn keyswitch_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        GLWECiphertext::keyswitch_scratch_space(
            module, n, basek, k_out, k_in, k_ksk, digits, rank_in, rank_out,
        )
    }

    pub fn keyswitch_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        GLWECiphertext::keyswitch_inplace_scratch_space(module, n, basek, k_out, k_ksk, digits, rank)
    }
}

impl<DataSelf: DataMut> GLWESwitchingKey<DataSelf> {
    pub fn keyswitch<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWESwitchingKey<DataLhs>,
        rhs: &GLWESwitchingKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B>,
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
        }

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                self.at_mut(row_j, col_i)
                    .keyswitch(module, &lhs.at(row_j, col_i), rhs, scratch);
            });
        });

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank_in()).for_each(|col_j| {
                self.at_mut(row_i, col_j).data.zero();
            });
        });
    }

    pub fn keyswitch_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GLWESwitchingKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
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

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                self.at_mut(row_j, col_i)
                    .keyswitch_inplace(module, rhs, scratch)
            });
        });
    }
}
