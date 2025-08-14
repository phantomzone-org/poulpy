use backend::hal::{
    api::{ScratchAvailable, TakeVecZnx, TakeVecZnxDft},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};
use sampling::source::Source;

use crate::{
    encryption::glwe_ct::glwe_encrypt_sk_internal,
    layouts::{GLWECiphertext, GLWEPlaintext, Infos, compressed::GLWECiphertextCompressed, prepared::GLWESecretExec},
};

use crate::trait_families::GLWEEncryptSkFamily;

impl GLWECiphertextCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize) -> usize
    where
        Module<B>: GLWEEncryptSkFamily<B>,
    {
        GLWECiphertext::encrypt_sk_scratch_space(module, n, basek, k)
    }
}

impl<D: DataMut> GLWECiphertextCompressed<D> {
    pub fn encrypt_sk<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &GLWEPlaintext<DataPt>,
        sk: &GLWESecretExec<DataSk, B>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEEncryptSkFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        self.encrypt_sk_internal(
            module,
            Some((pt, 0)),
            sk,
            seed_xa,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub(crate) fn encrypt_sk_internal<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: Option<(&GLWEPlaintext<DataPt>, usize)>,
        sk: &GLWESecretExec<DataSk, B>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEEncryptSkFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        let mut source_xa = Source::new(seed_xa);
        let cols: usize = self.rank() + 1;
        glwe_encrypt_sk_internal(
            module,
            self.basek(),
            self.k(),
            &mut self.data,
            cols,
            true,
            pt,
            sk,
            &mut source_xa,
            source_xe,
            sigma,
            scratch,
        );
        self.seed = seed_xa;
    }
}
