use backend::{Backend, Module, Scratch, VecZnxDftAllocBytes, ZnxZero};

use crate::{FourierGLWECiphertext, GLWEAutomorphismKey, GLWEAutomorphismKeyPrep, GLWESwitchingKey, GLWESwitchingKeyPrep, Infos};

impl GLWEAutomorphismKey<Vec<u8>> {
    pub fn keyswitch_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes,
    {
        GLWESwitchingKey::keyswitch_scratch_space(module, basek, k_out, k_in, k_ksk, digits, rank, rank)
    }

    pub fn keyswitch_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes,
    {
        GLWESwitchingKey::keyswitch_inplace_scratch_space(module, basek, k_out, k_ksk, digits, rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWEAutomorphismKey<DataSelf> {
    pub fn keyswitch<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWEAutomorphismKey<DataLhs>,
        rhs: &GLWESwitchingKeyPrep<DataRhs, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftAllocBytes,
    {
        self.key.keyswitch(module, &lhs.key, rhs, scratch);
    }

    pub fn keyswitch_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GLWEAutomorphismKeyPrep<DataRhs, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftAllocBytes,
    {
        self.key.keyswitch_inplace(module, &rhs.key, scratch);
    }
}

impl GLWESwitchingKey<Vec<u8>> {
    pub fn keyswitch_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes,
    {
        let tmp_in: usize = FourierGLWECiphertext::bytes_of(module, basek, k_in, rank_in);
        let tmp_out: usize = FourierGLWECiphertext::bytes_of(module, basek, k_out, rank_out);
        let ksk: usize =
            FourierGLWECiphertext::keyswitch_scratch_space(module, basek, k_out, k_in, k_ksk, digits, rank_in, rank_out);
        tmp_in + tmp_out + ksk
    }

    pub fn keyswitch_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes,
    {
        let tmp: usize = FourierGLWECiphertext::bytes_of(module, basek, k_out, rank);
        let ksk: usize = FourierGLWECiphertext::keyswitch_inplace_scratch_space(module, basek, k_out, k_ksk, digits, rank);
        tmp + ksk
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWESwitchingKey<DataSelf> {
    pub fn keyswitch<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWESwitchingKey<DataLhs>,
        rhs: &GLWESwitchingKeyPrep<DataRhs, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftAllocBytes,
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

    pub fn keyswitch_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GLWESwitchingKeyPrep<DataRhs, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftAllocBytes,
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
