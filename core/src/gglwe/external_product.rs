use backend::{Backend, Module, Scratch, ZnxZero};

use crate::{
    FourierGLWECiphertext, GLWEAutomorphismKey, GLWESwitchingKey, GetRow, Infos, ScratchCore, SetRow,
    ggsw::ciphertext_prep::GGSWCiphertextPrep,
};

impl GLWESwitchingKey<Vec<u8>> {
    pub fn external_product_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ggsw: usize,
        digits: usize,
        rank: usize,
    ) -> usize {
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
    ) -> usize {
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
    ) {
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

        let (mut tmp_in, scratch1) = scratch.tmp_fourier_glwe_ct(module, lhs.basek(), lhs.k(), lhs.rank());
        let (mut tmp_out, scratch2) = scratch1.tmp_fourier_glwe_ct(module, self.basek(), self.k(), self.rank());

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                lhs.get_row(module, row_j, col_i, &mut tmp_in);
                tmp_out.external_product(module, &tmp_in, rhs, scratch2);
                self.set_row(module, row_j, col_i, &tmp_out);
            });
        });

        tmp_out.data.zero();

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank_in()).for_each(|col_j| {
                self.set_row(module, row_i, col_j, &tmp_out);
            });
        });
    }

    pub fn external_product_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGSWCiphertextPrep<DataRhs, B>,
        scratch: &mut Scratch,
    ) {
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

        let (mut tmp, scratch1) = scratch.tmp_fourier_glwe_ct(module, self.basek(), self.k(), self.rank());
        println!("tmp: {}", tmp.size());
        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                self.get_row(module, row_j, col_i, &mut tmp);
                tmp.external_product_inplace(module, rhs, scratch1);
                self.set_row(module, row_j, col_i, &tmp);
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
    ) -> usize {
        GLWESwitchingKey::external_product_scratch_space(module, basek, k_out, k_in, ggsw_k, digits, rank)
    }

    pub fn external_product_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        ggsw_k: usize,
        digits: usize,
        rank: usize,
    ) -> usize {
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
    ) {
        self.key.external_product(module, &lhs.key, rhs, scratch);
    }

    pub fn external_product_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGSWCiphertextPrep<DataRhs, B>,
        scratch: &mut Scratch,
    ) {
        self.key.external_product_inplace(module, rhs, scratch);
    }
}
