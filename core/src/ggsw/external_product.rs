use backend::hal::{
    api::{ScratchAvailable, TakeVecZnxDft, ZnxZero},
    layouts::{Backend, Module, Scratch},
};

use crate::{GGSWCiphertext, GGSWCiphertextExec, GLWECiphertext, GLWEExternalProductFamily, Infos};

impl GGSWCiphertext<Vec<u8>> {
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
        Module<B>: GLWEExternalProductFamily<B>,
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
        Module<B>: GLWEExternalProductFamily<B>,
    {
        GLWECiphertext::external_product_inplace_scratch_space(module, basek, k_out, k_ggsw, digits, rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GGSWCiphertext<DataSelf> {
    pub fn external_product<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGSWCiphertext<DataLhs>,
        rhs: &GGSWCiphertextExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEExternalProductFamily<B>,
        Scratch<B>: ScratchAvailable + TakeVecZnxDft<B>,
    {
        #[cfg(debug_assertions)]
        {
            use crate::{GGSWCiphertext, Infos};

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

            assert!(
                scratch.available()
                    >= GGSWCiphertext::external_product_scratch_space(
                        module,
                        self.basek(),
                        self.k(),
                        lhs.k(),
                        rhs.k(),
                        rhs.digits(),
                        rhs.rank()
                    )
            )
        }

        let min_rows: usize = self.rows().min(lhs.rows());

        (0..self.rank() + 1).for_each(|col_i| {
            (0..min_rows).for_each(|row_j| {
                self.at_mut(row_j, col_i)
                    .external_product(module, &lhs.at(row_j, col_i), rhs, scratch);
            });
            (min_rows..self.rows()).for_each(|row_i| {
                self.at_mut(row_i, col_i).data.zero();
            });
        });
    }

    pub fn external_product_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGSWCiphertextExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEExternalProductFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank(),
                rhs.rank(),
                "ggsw_out rank: {} != ggsw_apply: {}",
                self.rank(),
                rhs.rank()
            );
        }

        (0..self.rank() + 1).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                self.at_mut(row_j, col_i)
                    .external_product_inplace(module, rhs, scratch);
            });
        });
    }
}
