use poulpy_hal::{
    api::{VecZnxDftAllocBytes, VecZnxNormalizeTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use crate::{
    encryption::{SIGMA, glwe_ct::GLWEEncryptSkInternal},
    layouts::{
        GLWECiphertext, GLWEInfos, GLWEPlaintext, GLWEPlaintextToRef, LWEInfos,
        compressed::{GLWECiphertextCompressed, GLWECiphertextCompressedToMut},
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

impl GLWECiphertextCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GLWEInfos,
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes,
    {
        GLWECiphertext::encrypt_sk_scratch_space(module, infos)
    }
}

pub trait GLWECompressedEncryptSk<B: Backend> {
    fn glwe_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GLWECiphertextCompressedToMut,
        P: GLWEPlaintextToRef,
        S: GLWESecretPreparedToRef<B>;
}

impl<B: Backend> GLWECompressedEncryptSk<B> for Module<B>
where
    Module<B>: GLWEEncryptSkInternal<B>,
{
    fn glwe_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GLWECiphertextCompressedToMut,
        P: GLWEPlaintextToRef,
        S: GLWESecretPreparedToRef<B>,
    {
        let res: &mut GLWECiphertextCompressed<&mut [u8]> = &mut res.to_mut();
        let mut source_xa: Source = Source::new(seed_xa);
        let cols: usize = (res.rank() + 1).into();

        self.glwe_encrypt_sk_internal(
            res.base2k().into(),
            res.k().into(),
            &mut res.data,
            cols,
            true,
            Some((pt, 0)),
            sk,
            &mut source_xa,
            source_xe,
            SIGMA,
            scratch,
        );

        res.seed = seed_xa;
    }
}

impl<D: DataMut> GLWECiphertextCompressed<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &GLWEPlaintext<DataPt>,
        sk: &GLWESecretPrepared<DataSk, B>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWECompressedEncryptSk<B>,
    {
        module.glwe_compressed_encrypt_sk(self, pt, sk, seed_xa, source_xe, scratch);
    }
}
