use backend::{Backend, Module, Scratch, VecZnxDftAllocBytes, ZnxZero};

use crate::{FourierGLWECiphertext, GLWEAutomorphismKey, GLWESwitchingKey, Infos, ggsw::ciphertext_prep::GGSWCiphertextPrep};

impl GLWESwitchingKey<Vec<u8>> {
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
        Module<B>: VecZnxDftAllocBytes,
    {
        let tmp_in: usize = FourierGLWECiphertext::bytes_of(module, basek, k_in, rank);
        let tmp_out: usize = FourierGLWECiphertext::bytes_of(module, basek, k_out, rank);
        let ggsw: usize = FourierGLWECiphertext::external_product_scratch_space(module, basek, k_out, k_in, k_ggsw, digits, rank);
        tmp_in + tmp_out + ggsw
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
        Module<B>: VecZnxDftAllocBytes,
    {
        let tmp: usize = FourierGLWECiphertext::bytes_of(module, basek, k_out, rank);
        let ggsw: usize =
            FourierGLWECiphertext::external_product_inplace_scratch_space(module, basek, k_out, k_ggsw, digits, rank);
        tmp + ggsw
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWESwitchingKey<DataSelf> {
    pub fn external_product<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWESwitchingKey<DataLhs>,
        rhs: &GGSWCiphertextPrep<DataRhs, B>,
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

    pub fn external_product_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGSWCiphertextPrep<DataRhs, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftAllocBytes,
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

impl GLWEAutomorphismKey<Vec<u8>> {
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
        Module<B>: VecZnxDftAllocBytes,
    {
        GLWESwitchingKey::external_product_scratch_space(module, basek, k_out, k_in, ggsw_k, digits, rank)
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
        Module<B>: VecZnxDftAllocBytes,
    {
        GLWESwitchingKey::external_product_inplace_scratch_space(module, basek, k_out, ggsw_k, digits, rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWEAutomorphismKey<DataSelf> {
    pub fn external_product<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GLWEAutomorphismKey<DataLhs>,
        rhs: &GGSWCiphertextPrep<DataRhs, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftAllocBytes,
    {
        self.key.external_product(module, &lhs.key, rhs, scratch);
    }

    pub fn external_product_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGSWCiphertextPrep<DataRhs, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftAllocBytes,
    {
        self.key.external_product_inplace(module, rhs, scratch);
    }
}
