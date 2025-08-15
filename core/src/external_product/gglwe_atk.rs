use backend::hal::{
    api::{ScratchAvailable, TakeVecZnxDft},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    layouts::{GGLWEAutomorphismKey, GGLWESwitchingKey, prepared::GGSWCiphertextPrepared},
    trait_families::GLWEExternalProductFamily,
};

impl GGLWEAutomorphismKey<Vec<u8>> {
    pub fn external_product_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k_out: usize,
        k_in: usize,
        ggsw_k: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEExternalProductFamily<B>,
    {
        GGLWESwitchingKey::external_product_scratch_space(module, n, basek, k_out, k_in, ggsw_k, digits, rank)
    }

    pub fn external_product_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k_out: usize,
        ggsw_k: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEExternalProductFamily<B>,
    {
        GGLWESwitchingKey::external_product_inplace_scratch_space(module, n, basek, k_out, ggsw_k, digits, rank)
    }
}

impl<DataSelf: DataMut> GGLWEAutomorphismKey<DataSelf> {
    pub fn external_product<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGLWEAutomorphismKey<DataLhs>,
        rhs: &GGSWCiphertextPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEExternalProductFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        self.key.external_product(module, &lhs.key, rhs, scratch);
    }

    pub fn external_product_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGSWCiphertextPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEExternalProductFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        self.key.external_product_inplace(module, rhs, scratch);
    }
}
