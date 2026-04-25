use poulpy_hal::{
    layouts::{Backend, HostDataMut, Module, ScalarZnxToBackendRef, ScratchArena},
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
    encryption::{
        GGLWECompressedEncryptSkDefault, GGLWEEncryptSkDefault, GGLWEToGGSWKeyCompressedEncryptSkDefault,
        GGLWEToGGSWKeyEncryptSkDefault, GGSWCompressedEncryptSkDefault, GGSWEncryptSkDefault,
        GLWEAutomorphismKeyCompressedEncryptSkDefault, GLWEAutomorphismKeyEncryptPkDefault, GLWEAutomorphismKeyEncryptSkDefault,
        GLWECompressedEncryptSkDefault, GLWEEncryptPkDefault, GLWEEncryptSkDefault, GLWEPublicKeyGenerateDefault,
        GLWESwitchingKeyCompressedEncryptSkDefault, GLWESwitchingKeyEncryptPkDefault, GLWESwitchingKeyEncryptSkDefault,
        GLWETensorKeyCompressedEncryptSkDefault, GLWETensorKeyEncryptSkDefault, GLWEToLWESwitchingKeyEncryptSkDefault,
        LWEEncryptSkDefault, LWESwitchingKeyEncryptDefault, LWEToGLWESwitchingKeyEncryptSkDefault,
    },
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToMut, GGLWEInfos, GGLWEToGGSWKeyCompressedToMut, GGLWEToGGSWKeyToMut, GGLWEToMut,
        GGSWCompressedSeedMut, GGSWCompressedToMut, GGSWInfos, GGSWToMut, GLWECompressedSeedMut, GLWECompressedToMut, GLWEInfos,
        GLWEPlaintextToBackendRef, GLWEPlaintextToRef, GLWESecretToRef, GLWESwitchingKeyDegreesMut, GLWEToMut, LWEInfos,
        LWEPlaintextToRef, LWESecretToRef, LWEToMut, SetGaloisElement,
        prepared::{GLWEPreparedToRef, GLWESecretPreparedToBackendRef},
    },
};

macro_rules! impl_encryption_delegate {
    ($trait:ty, $default:path, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend,
            Module<BE>: $default,
        {
            $($body)+
        }
    };
}

impl_encryption_delegate!(
    LWEEncryptSk<BE>,
    LWEEncryptSkDefault<BE>,
    fn lwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        <Module<BE> as LWEEncryptSkDefault<BE>>::lwe_encrypt_sk_tmp_bytes(self, infos)
    },
    fn lwe_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: LWEToMut,
        P: LWEPlaintextToRef,
        S: LWESecretToRef,
        E: EncryptionInfos,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        <Module<BE> as LWEEncryptSkDefault<BE>>::lwe_encrypt_sk(self, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    GLWEEncryptSk<BE>,
    GLWEEncryptSkDefault<BE>,
    fn glwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as GLWEEncryptSkDefault<BE>>::glwe_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef + GLWEPlaintextToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
    {
        <Module<BE> as GLWEEncryptSkDefault<BE>>::glwe_encrypt_sk(self, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    },
    fn glwe_encrypt_zero_sk<'s, R, E, S>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToMut,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
    {
        <Module<BE> as GLWEEncryptSkDefault<BE>>::glwe_encrypt_zero_sk(self, res, sk, enc_infos, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    GLWEEncryptPk<BE>,
    GLWEEncryptPkDefault<BE>,
    fn glwe_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as GLWEEncryptPkDefault<BE>>::glwe_encrypt_pk_tmp_bytes(self, infos)
    },
    fn glwe_encrypt_pk<'s, R, P, K, E>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        P: GLWEPlaintextToRef + GLWEPlaintextToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        <Module<BE> as GLWEEncryptPkDefault<BE>>::glwe_encrypt_pk(self, res, pt, pk, enc_infos, source_xu, source_xe, scratch)
    },
    fn glwe_encrypt_zero_pk<'s, R, K, E>(
        &self,
        res: &mut R,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        <Module<BE> as GLWEEncryptPkDefault<BE>>::glwe_encrypt_zero_pk(self, res, pk, enc_infos, source_xu, source_xe, scratch)
    }
);

impl_encryption_delegate!(
    GLWEPublicKeyGenerate<BE>,
    GLWEPublicKeyGenerateDefault<BE>,
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
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution,
    {
        <Module<BE> as GLWEPublicKeyGenerateDefault<BE>>::glwe_public_key_generate(self, res, sk, enc_infos, source_xe, source_xa)
    }
);

impl_encryption_delegate!(
    GGLWEEncryptSk<BE>,
    GGLWEEncryptSkDefault<BE>,
    fn gglwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GGLWEEncryptSkDefault<BE>>::gglwe_encrypt_sk_tmp_bytes(self, infos)
    },
    fn gglwe_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGLWEToMut,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        <Module<BE> as GGLWEEncryptSkDefault<BE>>::gglwe_encrypt_sk(self, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    GGSWEncryptSk<BE>,
    GGSWEncryptSkDefault<BE>,
    fn ggsw_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        <Module<BE> as GGSWEncryptSkDefault<BE>>::ggsw_encrypt_sk_tmp_bytes(self, infos)
    },
    fn ggsw_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToMut,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        <Module<BE> as GGSWEncryptSkDefault<BE>>::ggsw_encrypt_sk(self, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    GGLWEToGGSWKeyEncryptSk<BE>,
    GGLWEToGGSWKeyEncryptSkDefault<BE>,
    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GGLWEToGGSWKeyEncryptSkDefault<BE>>::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(self, infos)
    },
    fn gglwe_to_ggsw_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToGGSWKeyToMut,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        <Module<BE> as GGLWEToGGSWKeyEncryptSkDefault<BE>>::gglwe_to_ggsw_key_encrypt_sk(
            self, res, sk, enc_infos, source_xe, source_xa, scratch,
        )
    }
);

impl_encryption_delegate!(
    GLWESwitchingKeyEncryptSk<BE>,
    GLWESwitchingKeyEncryptSkDefault<BE>,
    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWESwitchingKeyEncryptSkDefault<BE>>::glwe_switching_key_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_switching_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToRef,
        S2: GLWESecretToRef,
    {
        <Module<BE> as GLWESwitchingKeyEncryptSkDefault<BE>>::glwe_switching_key_encrypt_sk(
            self, res, sk_in, sk_out, enc_infos, source_xe, source_xa, scratch,
        )
    }
);

impl_encryption_delegate!(
    GLWESwitchingKeyEncryptPk<BE>,
    GLWESwitchingKeyEncryptPkDefault<BE>,
    fn glwe_switching_key_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWESwitchingKeyEncryptPkDefault<BE>>::glwe_switching_key_encrypt_pk_tmp_bytes(self, infos)
    }
);

impl_encryption_delegate!(
    GLWETensorKeyEncryptSk<BE>,
    GLWETensorKeyEncryptSkDefault<BE>,
    fn glwe_tensor_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWETensorKeyEncryptSkDefault<BE>>::glwe_tensor_key_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_tensor_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToMut + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        <Module<BE> as GLWETensorKeyEncryptSkDefault<BE>>::glwe_tensor_key_encrypt_sk(
            self, res, sk, enc_infos, source_xe, source_xa, scratch,
        )
    }
);

impl_encryption_delegate!(
    GLWEToLWESwitchingKeyEncryptSk<BE>,
    GLWEToLWESwitchingKeyEncryptSkDefault<BE>,
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWEToLWESwitchingKeyEncryptSkDefault<BE>>::glwe_to_lwe_key_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_to_lwe_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretToRef,
        E: EncryptionInfos,
        R: GGLWEToMut + GGLWEInfos,
    {
        <Module<BE> as GLWEToLWESwitchingKeyEncryptSkDefault<BE>>::glwe_to_lwe_key_encrypt_sk(
            self, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch,
        )
    }
);

impl_encryption_delegate!(
    LWESwitchingKeyEncrypt<BE>,
    LWESwitchingKeyEncryptDefault<BE>,
    fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as LWESwitchingKeyEncryptDefault<BE>>::lwe_switching_key_encrypt_sk_tmp_bytes(self, infos)
    },
    fn lwe_switching_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToRef,
        S2: LWESecretToRef,
    {
        <Module<BE> as LWESwitchingKeyEncryptDefault<BE>>::lwe_switching_key_encrypt_sk(
            self, res, sk_lwe_in, sk_lwe_out, enc_infos, source_xe, source_xa, scratch,
        )
    }
);

impl_encryption_delegate!(
    LWEToGLWESwitchingKeyEncryptSk<BE>,
    LWEToGLWESwitchingKeyEncryptSkDefault<BE>,
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as LWEToGLWESwitchingKeyEncryptSkDefault<BE>>::lwe_to_glwe_key_encrypt_sk_tmp_bytes(self, infos)
    },
    fn lwe_to_glwe_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos,
        R: GGLWEToMut + GGLWEInfos,
    {
        <Module<BE> as LWEToGLWESwitchingKeyEncryptSkDefault<BE>>::lwe_to_glwe_key_encrypt_sk(
            self, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch,
        )
    }
);

impl_encryption_delegate!(
    GLWEAutomorphismKeyEncryptSk<BE>,
    GLWEAutomorphismKeyEncryptSkDefault<BE>,
    fn glwe_automorphism_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismKeyEncryptSkDefault<BE>>::glwe_automorphism_key_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_automorphism_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef,
    {
        <Module<BE> as GLWEAutomorphismKeyEncryptSkDefault<BE>>::glwe_automorphism_key_encrypt_sk(
            self, res, p, sk, enc_infos, source_xe, source_xa, scratch,
        )
    }
);

impl_encryption_delegate!(
    GLWEAutomorphismKeyEncryptPk<BE>,
    GLWEAutomorphismKeyEncryptPkDefault<BE>,
    fn glwe_automorphism_key_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismKeyEncryptPkDefault<BE>>::glwe_automorphism_key_encrypt_pk_tmp_bytes(self, infos)
    }
);

impl_encryption_delegate!(
    GLWECompressedEncryptSk<BE>,
    GLWECompressedEncryptSkDefault<BE>,
    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as GLWECompressedEncryptSkDefault<BE>>::glwe_compressed_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_compressed_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWECompressedToMut + GLWECompressedSeedMut,
        P: GLWEPlaintextToRef + GLWEPlaintextToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE::BufMut<'s>: HostDataMut,
    {
        <Module<BE> as GLWECompressedEncryptSkDefault<BE>>::glwe_compressed_encrypt_sk(
            self, res, pt, sk, seed_xa, enc_infos, source_xe, scratch,
        )
    }
);

impl_encryption_delegate!(
    GGLWECompressedEncryptSk<BE>,
    GGLWECompressedEncryptSkDefault<BE>,
    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GGLWECompressedEncryptSkDefault<BE>>::gglwe_compressed_encrypt_sk_tmp_bytes(self, infos)
    },
    fn gglwe_compressed_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        <Module<BE> as GGLWECompressedEncryptSkDefault<BE>>::gglwe_compressed_encrypt_sk(
            self, res, pt, sk, seed, enc_infos, source_xe, scratch,
        )
    }
);

impl_encryption_delegate!(
    GGSWCompressedEncryptSk<BE>,
    GGSWCompressedEncryptSkDefault<BE>,
    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        <Module<BE> as GGSWCompressedEncryptSkDefault<BE>>::ggsw_compressed_encrypt_sk_tmp_bytes(self, infos)
    },
    fn ggsw_compressed_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWCompressedToMut + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        <Module<BE> as GGSWCompressedEncryptSkDefault<BE>>::ggsw_compressed_encrypt_sk(
            self, res, pt, sk, seed_xa, enc_infos, source_xe, scratch,
        )
    }
);

impl_encryption_delegate!(
    GGLWEToGGSWKeyCompressedEncryptSk<BE>,
    GGLWEToGGSWKeyCompressedEncryptSkDefault<BE>,
    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GGLWEToGGSWKeyCompressedEncryptSkDefault<BE>>::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(self, infos)
    },
    fn gglwe_to_ggsw_key_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToGGSWKeyCompressedToMut + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        <Module<BE> as GGLWEToGGSWKeyCompressedEncryptSkDefault<BE>>::gglwe_to_ggsw_key_encrypt_sk(
            self, res, sk, seed_xa, enc_infos, source_xe, scratch,
        )
    }
);

impl_encryption_delegate!(
    GLWEAutomorphismKeyCompressedEncryptSk<BE>,
    GLWEAutomorphismKeyCompressedEncryptSkDefault<BE>,
    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismKeyCompressedEncryptSkDefault<BE>>::glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(
            self, infos,
        )
    },
    fn glwe_automorphism_key_compressed_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismKeyCompressedEncryptSkDefault<BE>>::glwe_automorphism_key_compressed_encrypt_sk(
            self, res, p, sk, seed_xa, enc_infos, source_xe, scratch,
        )
    }
);

impl_encryption_delegate!(
    GLWESwitchingKeyCompressedEncryptSk<BE>,
    GLWESwitchingKeyCompressedEncryptSkDefault<BE>,
    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWESwitchingKeyCompressedEncryptSkDefault<BE>>::glwe_switching_key_compressed_encrypt_sk_tmp_bytes(
            self, infos,
        )
    },
    fn glwe_switching_key_compressed_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToRef,
        S2: GLWESecretToRef,
    {
        <Module<BE> as GLWESwitchingKeyCompressedEncryptSkDefault<BE>>::glwe_switching_key_compressed_encrypt_sk(
            self, res, sk_in, sk_out, seed_xa, enc_infos, source_xe, scratch,
        )
    }
);

impl_encryption_delegate!(
    GLWETensorKeyCompressedEncryptSk<BE>,
    GLWETensorKeyCompressedEncryptSkDefault<BE>,
    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWETensorKeyCompressedEncryptSkDefault<BE>>::glwe_tensor_key_compressed_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_tensor_key_compressed_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToMut + GGLWEInfos + GGLWECompressedSeedMut,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        <Module<BE> as GLWETensorKeyCompressedEncryptSkDefault<BE>>::glwe_tensor_key_compressed_encrypt_sk(
            self, res, sk, seed_xa, enc_infos, source_xe, scratch,
        )
    }
);
