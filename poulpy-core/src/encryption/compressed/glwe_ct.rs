use poulpy_hal::{
    layouts::{Backend, DataMut, Module, Scratch},
    source::Source,
};

use crate::{
    encryption::{
        SIGMA,
        glwe_ct::{GLWEEncryptSk, GLWEEncryptSkInternal},
    },
    layouts::{
        GLWEInfos, GLWEPlaintextToRef, LWEInfos,
        compressed::{GLWECompressed, GLWECompressedToMut},
        prepared::GLWESecretPreparedToRef,
    },
};

impl GLWECompressed<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GLWEInfos,
        M: GLWECompressedEncryptSk<BE>,
    {
        module.glwe_compressed_encrypt_sk_tmp_bytes(infos)
    }
}

impl<D: DataMut> GLWECompressed<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<M, P, S, BE: Backend>(
        &mut self,
        module: &M,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        M: GLWECompressedEncryptSk<BE>,
        P: GLWEPlaintextToRef,
        S: GLWESecretPreparedToRef<BE>,
    {
        module.glwe_compressed_encrypt_sk(self, pt, sk, seed_xa, source_xe, scratch);
    }
}

pub trait GLWECompressedEncryptSk<BE: Backend> {
    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWECompressedToMut,
        P: GLWEPlaintextToRef,
        S: GLWESecretPreparedToRef<BE>;
}

impl<BE: Backend> GLWECompressedEncryptSk<BE> for Module<BE>
where
    Self: GLWEEncryptSkInternal<BE> + GLWEEncryptSk<BE>,
{
    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        self.glwe_encrypt_sk_tmp_bytes(infos)
    }

    fn glwe_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWECompressedToMut,
        P: GLWEPlaintextToRef,
        S: GLWESecretPreparedToRef<BE>,
    {
        let res: &mut GLWECompressed<&mut [u8]> = &mut res.to_mut();
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
