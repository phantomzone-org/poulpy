#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, Scratch},
    source::Source,
};

pub use crate::api::GLWECompressedEncryptSk;
use crate::{
    EncryptionInfos,
    encryption::{GLWEEncryptSk, GLWEEncryptSkInternal},
    layouts::{
        GLWECompressedSeedMut, GLWEInfos, GLWEPlaintextToRef, LWEInfos,
        compressed::{GLWECompressed, GLWECompressedToMut},
        prepared::GLWESecretPreparedToRef,
    },
};

#[doc(hidden)]
pub trait GLWECompressedEncryptSkDefault<BE: Backend> {
    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_compressed_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWECompressedToMut + GLWECompressedSeedMut,
        P: GLWEPlaintextToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;
}

impl<BE: Backend> GLWECompressedEncryptSkDefault<BE> for Module<BE>
where
    Self: GLWEEncryptSkInternal<BE> + GLWEEncryptSk<BE>,
    Scratch<BE>: ScratchAvailable,
{
    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());
        let lvl_0: usize = self.glwe_encrypt_sk_tmp_bytes(infos);
        lvl_0
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_compressed_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWECompressedToMut + GLWECompressedSeedMut,
        P: GLWEPlaintextToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>,
    {
        {
            let res: &mut GLWECompressed<&mut [u8]> = &mut res.to_mut();
            let mut source_xa: Source = Source::new(seed_xa);
            let cols: usize = (res.rank() + 1).into();
            assert!(
                scratch.available() >= self.glwe_compressed_encrypt_sk_tmp_bytes(res),
                "scratch.available(): {} < GLWECompressedEncryptSk::glwe_compressed_encrypt_sk_tmp_bytes: {}",
                scratch.available(),
                self.glwe_compressed_encrypt_sk_tmp_bytes(res)
            );

            self.glwe_encrypt_sk_internal(
                res.base2k().into(),
                &mut res.data,
                cols,
                true,
                Some((pt, 0)),
                sk,
                enc_infos,
                source_xe,
                &mut source_xa,
                scratch,
            );
        }

        res.seed_mut().copy_from_slice(&seed_xa);
    }
}
