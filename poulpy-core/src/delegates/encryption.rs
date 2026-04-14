use poulpy_hal::{
    layouts::{Backend, Module, ScalarZnxToRef, Scratch},
    source::Source,
};

use crate::{
    GetDistribution, GetDistributionMut,
    api::{
        EncryptionInfos, GGLWECompressedEncryptSk, GGLWEEncryptSk, GGLWEToGGSWKeyCompressedEncryptSk, GGLWEToGGSWKeyEncryptSk,
        GGSWCompressedEncryptSk, GGSWEncryptSk, GLWEAutomorphismKeyCompressedEncryptSk, GLWEAutomorphismKeyEncryptPk,
        GLWEAutomorphismKeyEncryptSk, GLWECompressedEncryptSk, GLWEEncryptPk, GLWEEncryptSk, GLWEPublicKeyGenerate,
        GLWESwitchingKeyCompressedEncryptSk, GLWESwitchingKeyEncryptPk, GLWESwitchingKeyEncryptSk,
        GLWETensorKeyCompressedEncryptSk, GLWETensorKeyEncryptSk, GLWEToLWESwitchingKeyEncryptSk, LWEEncryptSk,
        LWESwitchingKeyEncrypt, LWEToGLWESwitchingKeyEncryptSk,
    },
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToMut, GGLWEInfos, GGLWEToGGSWKeyCompressedToMut, GGLWEToGGSWKeyToMut, GGLWEToMut,
        GGSWCompressedSeedMut, GGSWCompressedToMut, GGSWInfos, GGSWToMut, GLWECompressedSeedMut, GLWECompressedToMut, GLWEInfos,
        GLWEPlaintextToRef, GLWEPreparedToRef, GLWESecretPreparedToRef, GLWESecretToRef, GLWESwitchingKeyDegreesMut, GLWEToMut,
        LWEInfos, LWEPlaintextToRef, LWESecretToRef, LWEToMut, SetGaloisElement,
    },
    oep::CoreImpl,
};

impl<BE> LWEEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn lwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        BE::lwe_encrypt_sk_tmp_bytes(self, infos)
    }

    fn lwe_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: LWEToMut,
        P: LWEPlaintextToRef,
        S: LWESecretToRef,
        E: EncryptionInfos,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::lwe_encrypt_sk(self, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }
}

impl<BE> GLWEEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_encrypt_sk_tmp_bytes(self, infos)
    }

    fn glwe_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>,
    {
        BE::glwe_encrypt_sk(self, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn glwe_encrypt_zero_sk<R, E, S>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>,
    {
        BE::glwe_encrypt_zero_sk(self, res, sk, enc_infos, source_xe, source_xa, scratch)
    }
}

impl<BE> GLWEEncryptPk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_encrypt_pk_tmp_bytes(self, infos)
    }

    fn glwe_encrypt_pk<R, P, K, E>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        P: GLWEPlaintextToRef + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos,
    {
        BE::glwe_encrypt_pk(self, res, pt, pk, enc_infos, source_xu, source_xe, scratch)
    }

    fn glwe_encrypt_zero_pk<R, K, E>(
        &self,
        res: &mut R,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos,
    {
        BE::glwe_encrypt_zero_pk(self, res, pk, enc_infos, source_xu, source_xe, scratch)
    }
}

impl<BE> GLWEPublicKeyGenerate<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_public_key_generate<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
    ) where
        R: GLWEToMut + GetDistributionMut + GLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE> + GetDistribution,
    {
        BE::glwe_public_key_generate(self, res, sk, enc_infos, source_xe, source_xa)
    }
}

impl<BE> GGLWEEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn gglwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::gglwe_encrypt_sk_tmp_bytes(self, infos)
    }

    fn gglwe_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>,
    {
        BE::gglwe_encrypt_sk(self, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }
}

impl<BE> GGSWEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn ggsw_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        BE::ggsw_encrypt_sk_tmp_bytes(self, infos)
    }

    fn ggsw_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>,
    {
        BE::ggsw_encrypt_sk(self, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }
}

impl<BE> GGLWEToGGSWKeyEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(self, infos)
    }

    fn gglwe_to_ggsw_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToGGSWKeyToMut,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        BE::gglwe_to_ggsw_key_encrypt_sk(self, res, sk, enc_infos, source_xe, source_xa, scratch)
    }
}

impl<BE> GLWESwitchingKeyEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_switching_key_encrypt_sk_tmp_bytes(self, infos)
    }

    fn glwe_switching_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToRef,
        S2: GLWESecretToRef,
    {
        BE::glwe_switching_key_encrypt_sk(self, res, sk_in, sk_out, enc_infos, source_xe, source_xa, scratch)
    }
}

impl<BE> GLWESwitchingKeyEncryptPk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_switching_key_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_switching_key_encrypt_pk_tmp_bytes(self, infos)
    }
}

impl<BE> GLWETensorKeyEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_tensor_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_tensor_key_encrypt_sk_tmp_bytes(self, infos)
    }

    fn glwe_tensor_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        BE::glwe_tensor_key_encrypt_sk(self, res, sk, enc_infos, source_xe, source_xa, scratch)
    }
}

impl<BE> GLWEToLWESwitchingKeyEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_to_lwe_key_encrypt_sk_tmp_bytes(self, infos)
    }

    fn glwe_to_lwe_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretToRef,
        E: EncryptionInfos,
        R: GGLWEToMut + GGLWEInfos,
    {
        BE::glwe_to_lwe_key_encrypt_sk(self, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch)
    }
}

impl<BE> LWESwitchingKeyEncrypt<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::lwe_switching_key_encrypt_sk_tmp_bytes(self, infos)
    }

    fn lwe_switching_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToRef,
        S2: LWESecretToRef,
    {
        BE::lwe_switching_key_encrypt_sk(self, res, sk_lwe_in, sk_lwe_out, enc_infos, source_xe, source_xa, scratch)
    }
}

impl<BE> LWEToGLWESwitchingKeyEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::lwe_to_glwe_key_encrypt_sk_tmp_bytes(self, infos)
    }

    fn lwe_to_glwe_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretPreparedToRef<BE>,
        E: EncryptionInfos,
        R: GGLWEToMut + GGLWEInfos,
    {
        BE::lwe_to_glwe_key_encrypt_sk(self, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch)
    }
}

impl<BE> GLWEAutomorphismKeyEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_automorphism_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_automorphism_key_encrypt_sk_tmp_bytes(self, infos)
    }

    fn glwe_automorphism_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef,
    {
        BE::glwe_automorphism_key_encrypt_sk(self, res, p, sk, enc_infos, source_xe, source_xa, scratch)
    }
}

impl<BE> GLWEAutomorphismKeyEncryptPk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_automorphism_key_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_automorphism_key_encrypt_pk_tmp_bytes(self, infos)
    }
}

impl<BE> GLWECompressedEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_compressed_encrypt_sk_tmp_bytes(self, infos)
    }

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
        BE::glwe_compressed_encrypt_sk(self, res, pt, sk, seed_xa, enc_infos, source_xe, scratch)
    }
}

impl<BE> GGLWECompressedEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::gglwe_compressed_encrypt_sk_tmp_bytes(self, infos)
    }

    fn gglwe_compressed_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>,
    {
        BE::gglwe_compressed_encrypt_sk(self, res, pt, sk, seed, enc_infos, source_xe, scratch)
    }
}

impl<BE> GGSWCompressedEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        BE::ggsw_compressed_encrypt_sk_tmp_bytes(self, infos)
    }

    fn ggsw_compressed_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWCompressedToMut + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>,
    {
        BE::ggsw_compressed_encrypt_sk(self, res, pt, sk, seed_xa, enc_infos, source_xe, scratch)
    }
}

impl<BE> GGLWEToGGSWKeyCompressedEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes(self, infos)
    }

    fn gglwe_to_ggsw_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToGGSWKeyCompressedToMut + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        BE::gglwe_to_ggsw_key_compressed_encrypt_sk(self, res, sk, seed_xa, enc_infos, source_xe, scratch)
    }
}

impl<BE> GLWEAutomorphismKeyCompressedEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(self, infos)
    }

    fn glwe_automorphism_key_compressed_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GLWEInfos,
    {
        BE::glwe_automorphism_key_compressed_encrypt_sk(self, res, p, sk, seed_xa, enc_infos, source_xe, scratch)
    }
}

impl<BE> GLWESwitchingKeyCompressedEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_switching_key_compressed_encrypt_sk_tmp_bytes(self, infos)
    }

    fn glwe_switching_key_compressed_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToRef,
        S2: GLWESecretToRef,
    {
        BE::glwe_switching_key_compressed_encrypt_sk(self, res, sk_in, sk_out, seed_xa, enc_infos, source_xe, scratch)
    }
}

impl<BE> GLWETensorKeyCompressedEncryptSk<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_tensor_key_compressed_encrypt_sk_tmp_bytes(self, infos)
    }

    fn glwe_tensor_key_compressed_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWEInfos + GGLWECompressedSeedMut,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        BE::glwe_tensor_key_compressed_encrypt_sk(self, res, sk, seed_xa, enc_infos, source_xe, scratch)
    }
}
