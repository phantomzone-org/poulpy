use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, ScratchTakeBasic, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare, VecZnxDftBytesOf, VecZnxNormalizeTmpBytes, VecZnxSwitchRing},
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, Scratch},
    source::Source,
};

use crate::{
    ScratchTakeCore,
    encryption::compressed::gglwe_ct::GGLWECompressedEncryptSk,
    layouts::{
        GGLWE, GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretToRef, LWEInfos, RingDegree,
        compressed::{GLWESwitchingKeyCompressed, GLWESwitchingKeyCompressedToMut},
        prepared::GLWESecretPrepared,
    },
};

impl GLWESwitchingKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: ModuleN + SvpPPolAlloc<B> + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf + VecZnxNormalizeTmpBytes + SvpPPolBytesOf,
    {
        (GGLWE::encrypt_sk_tmp_bytes(module, infos) | ScalarZnx::bytes_of(module.n(), 1))
            + ScalarZnx::bytes_of(module.n(), infos.rank_in().into())
            + GLWESecretPrepared::bytes_of(module, infos.rank_out())
    }
}

impl<DataSelf: DataMut> GLWESwitchingKeyCompressed<DataSelf> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataSkIn: DataRef, DataSkOut: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_in: &GLWESecret<DataSkIn>,
        sk_out: &GLWESecret<DataSkOut>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GGLWEKeyCompressedEncryptSk<B>,
    {
        module.gglwe_key_compressed_encrypt_sk(self, sk_in, sk_out, seed_xa, source_xe, scratch);
    }
}

pub trait GGLWEKeyCompressedEncryptSk<B: Backend> {
    fn gglwe_key_compressed_encrypt_sk<R, SI, SO>(
        &self,
        res: &mut R,
        sk_in: &SI,
        sk_out: &SO,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GLWESwitchingKeyCompressedToMut,
        SI: GLWESecretToRef,
        SO: GLWESecretToRef;
}

impl<B: Backend> GGLWEKeyCompressedEncryptSk<B> for Module<B>
where
    Module<B>: ModuleN
        + GGLWECompressedEncryptSk<B>
        + SvpPPolBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxSwitchRing
        + SvpPrepare<B>
        + SvpPPolAlloc<B>,
    Scratch<B>: ScratchAvailable + ScratchTakeBasic + ScratchTakeCore<B>,
{
    fn gglwe_key_compressed_encrypt_sk<R, SI, SO>(
        &self,
        res: &mut R,
        sk_in: &SI,
        sk_out: &SO,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GLWESwitchingKeyCompressedToMut,
        SI: GLWESecretToRef,
        SO: GLWESecretToRef,
    {
        let res: &mut GLWESwitchingKeyCompressed<&mut [u8]> = &mut res.to_mut();
        let sk_in: &GLWESecret<&[u8]> = &sk_in.to_ref();
        let sk_out: &GLWESecret<&[u8]> = &sk_out.to_ref();

        #[cfg(debug_assertions)]
        {
            use crate::layouts::GLWESwitchingKey;

            assert!(sk_in.n().0 <= self.n() as u32);
            assert!(sk_out.n().0 <= self.n() as u32);
            assert!(
                scratch.available() >= GLWESwitchingKey::encrypt_sk_tmp_bytes(self, res),
                "scratch.available()={} < GLWESwitchingKey::encrypt_sk_tmp_bytes={}",
                scratch.available(),
                GLWESwitchingKey::encrypt_sk_tmp_bytes(self, res)
            )
        }

        let n: usize = sk_in.n().max(sk_out.n()).into();

        let (mut sk_in_tmp, scratch_1) = scratch.take_scalar_znx(self, sk_in.rank().into());
        (0..sk_in.rank().into()).for_each(|i| {
            self.vec_znx_switch_ring(
                &mut sk_in_tmp.as_vec_znx_mut(),
                i,
                &sk_in.data.as_vec_znx(),
                i,
            );
        });

        let (mut sk_out_tmp, scratch_2) = scratch_1.take_glwe_secret_prepared(self, sk_out.rank());
        {
            let (mut tmp, _) = scratch_2.take_scalar_znx(self, 1);
            (0..sk_out.rank().into()).for_each(|i| {
                self.vec_znx_switch_ring(&mut tmp.as_vec_znx_mut(), 0, &sk_out.data.as_vec_znx(), i);
                self.svp_prepare(&mut sk_out_tmp.data, i, &tmp, 0);
            });
        }

        self.gglwe_compressed_encrypt_sk(
            &mut res.key,
            &sk_in_tmp,
            &sk_out_tmp,
            seed_xa,
            source_xe,
            scratch_2,
        );
        res.sk_in_n = sk_in.n().into();
        res.sk_out_n = sk_out.n().into();
    }
}
