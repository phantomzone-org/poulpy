#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    layouts::{Backend, Module, ScalarZnxToRef, Scratch},
    source::Source,
};

use crate::{
    EncryptionInfos, GetDistribution, GetDistributionMut, ScratchTakeCore,
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
        GLWEPlaintextToRef, GLWEPreparedToRef, GLWESecretPreparedToRef, GLWESecretToRef, GLWESwitchingKeyDegreesMut, GLWEToMut,
        LWEInfos, LWEPlaintextToRef, LWESecretToRef, LWEToMut, SetGaloisElement,
    },
};

#[doc(hidden)]
pub trait CoreEncryptionDefaults<BE: Backend>: Backend {
    fn lwe_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_encrypt_sk_default<R, P, S, E>(
        module: &Module<BE>,
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
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_sk_default<R, P, S, E>(
        module: &Module<BE>,
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
        S: GLWESecretPreparedToRef<BE>;

    fn glwe_encrypt_zero_sk_default<R, E, S>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;

    fn glwe_encrypt_pk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_pk_default<R, P, K, E>(
        module: &Module<BE>,
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
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos;

    fn glwe_encrypt_zero_pk_default<R, K, E>(
        module: &Module<BE>,
        res: &mut R,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos;

    fn glwe_public_key_generate_default<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
    ) where
        R: GLWEToMut + GetDistributionMut + GLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE> + GetDistribution;

    fn gglwe_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_encrypt_sk_default<R, P, S, E>(
        module: &Module<BE>,
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
        S: GLWESecretPreparedToRef<BE>;

    fn ggsw_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_encrypt_sk_default<R, P, S, E>(
        module: &Module<BE>,
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
        S: GLWESecretPreparedToRef<BE>;

    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_encrypt_sk_default<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToGGSWKeyToMut,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;

    fn glwe_switching_key_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_encrypt_sk_default<R, S1, S2, E>(
        module: &Module<BE>,
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
        S2: GLWESecretToRef;

    fn glwe_switching_key_encrypt_pk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_encrypt_sk_default<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;

    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_to_lwe_key_encrypt_sk_default<R, S1, S2, E>(
        module: &Module<BE>,
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
        R: GGLWEToMut + GGLWEInfos;

    fn lwe_switching_key_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_switching_key_encrypt_sk_default<R, S1, S2, E>(
        module: &Module<BE>,
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
        S2: LWESecretToRef;

    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_to_glwe_key_encrypt_sk_default<R, S1, S2, E>(
        module: &Module<BE>,
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
        R: GGLWEToMut + GGLWEInfos;

    fn glwe_automorphism_key_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_encrypt_sk_default<R, S, E>(
        module: &Module<BE>,
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
        S: GLWESecretToRef;

    fn glwe_automorphism_key_encrypt_pk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_compressed_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_compressed_encrypt_sk_default<R, P, S, E>(
        module: &Module<BE>,
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

    fn gglwe_compressed_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_compressed_encrypt_sk_default<R, P, S, E>(
        module: &Module<BE>,
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
        S: GLWESecretPreparedToRef<BE>;

    fn ggsw_compressed_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_compressed_encrypt_sk_default<R, P, S, E>(
        module: &Module<BE>,
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
        S: GLWESecretPreparedToRef<BE>;

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_default<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToGGSWKeyCompressedToMut + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;

    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_compressed_encrypt_sk_default<R, S, E>(
        module: &Module<BE>,
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
        S: GLWESecretToRef + GLWEInfos;

    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_compressed_encrypt_sk_default<R, S1, S2, E>(
        module: &Module<BE>,
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
        S2: GLWESecretToRef;

    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_compressed_encrypt_sk_default<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWEInfos + GGLWECompressedSeedMut,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;
}

impl<BE: Backend> CoreEncryptionDefaults<BE> for BE
where
    Module<BE>: LWEEncryptSkDefault<BE>
        + GLWEEncryptSkDefault<BE>
        + GLWEEncryptPkDefault<BE>
        + GLWEPublicKeyGenerateDefault<BE>
        + GGLWEEncryptSkDefault<BE>
        + GGSWEncryptSkDefault<BE>
        + GGLWEToGGSWKeyEncryptSkDefault<BE>
        + GLWESwitchingKeyEncryptSkDefault<BE>
        + GLWESwitchingKeyEncryptPkDefault<BE>
        + GLWETensorKeyEncryptSkDefault<BE>
        + GLWEToLWESwitchingKeyEncryptSkDefault<BE>
        + LWESwitchingKeyEncryptDefault<BE>
        + LWEToGLWESwitchingKeyEncryptSkDefault<BE>
        + GLWEAutomorphismKeyEncryptSkDefault<BE>
        + GLWEAutomorphismKeyEncryptPkDefault<BE>
        + GLWECompressedEncryptSkDefault<BE>
        + GGLWECompressedEncryptSkDefault<BE>
        + GGSWCompressedEncryptSkDefault<BE>
        + GGLWEToGGSWKeyCompressedEncryptSkDefault<BE>
        + GLWEAutomorphismKeyCompressedEncryptSkDefault<BE>
        + GLWESwitchingKeyCompressedEncryptSkDefault<BE>
        + GLWETensorKeyCompressedEncryptSkDefault<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn lwe_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        <Module<BE> as LWEEncryptSkDefault<BE>>::lwe_encrypt_sk_tmp_bytes(module, infos)
    }

    fn lwe_encrypt_sk_default<R, P, S, E>(
        module: &Module<BE>,
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
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as LWEEncryptSkDefault<BE>>::lwe_encrypt_sk(module, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn glwe_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as GLWEEncryptSkDefault<BE>>::glwe_encrypt_sk_tmp_bytes(module, infos)
    }

    fn glwe_encrypt_sk_default<R, P, S, E>(
        module: &Module<BE>,
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
        <Module<BE> as GLWEEncryptSkDefault<BE>>::glwe_encrypt_sk(module, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn glwe_encrypt_zero_sk_default<R, E, S>(
        module: &Module<BE>,
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
        <Module<BE> as GLWEEncryptSkDefault<BE>>::glwe_encrypt_zero_sk(module, res, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn glwe_encrypt_pk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as GLWEEncryptPkDefault<BE>>::glwe_encrypt_pk_tmp_bytes(module, infos)
    }

    fn glwe_encrypt_pk_default<R, P, K, E>(
        module: &Module<BE>,
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
        <Module<BE> as GLWEEncryptPkDefault<BE>>::glwe_encrypt_pk(module, res, pt, pk, enc_infos, source_xu, source_xe, scratch)
    }

    fn glwe_encrypt_zero_pk_default<R, K, E>(
        module: &Module<BE>,
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
        <Module<BE> as GLWEEncryptPkDefault<BE>>::glwe_encrypt_zero_pk(module, res, pk, enc_infos, source_xu, source_xe, scratch)
    }

    fn glwe_public_key_generate_default<R, S, E>(
        module: &Module<BE>,
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
        <Module<BE> as GLWEPublicKeyGenerateDefault<BE>>::glwe_public_key_generate(
            module, res, sk, enc_infos, source_xe, source_xa,
        )
    }

    fn gglwe_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GGLWEEncryptSkDefault<BE>>::gglwe_encrypt_sk_tmp_bytes(module, infos)
    }

    fn gglwe_encrypt_sk_default<R, P, S, E>(
        module: &Module<BE>,
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
        <Module<BE> as GGLWEEncryptSkDefault<BE>>::gglwe_encrypt_sk(module, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn ggsw_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        <Module<BE> as GGSWEncryptSkDefault<BE>>::ggsw_encrypt_sk_tmp_bytes(module, infos)
    }

    fn ggsw_encrypt_sk_default<R, P, S, E>(
        module: &Module<BE>,
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
        <Module<BE> as GGSWEncryptSkDefault<BE>>::ggsw_encrypt_sk(module, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GGLWEToGGSWKeyEncryptSkDefault<BE>>::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(module, infos)
    }

    fn gglwe_to_ggsw_key_encrypt_sk_default<R, S, E>(
        module: &Module<BE>,
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
        <Module<BE> as GGLWEToGGSWKeyEncryptSkDefault<BE>>::gglwe_to_ggsw_key_encrypt_sk(
            module, res, sk, enc_infos, source_xe, source_xa, scratch,
        )
    }

    fn glwe_switching_key_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWESwitchingKeyEncryptSkDefault<BE>>::glwe_switching_key_encrypt_sk_tmp_bytes(module, infos)
    }

    fn glwe_switching_key_encrypt_sk_default<R, S1, S2, E>(
        module: &Module<BE>,
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
        <Module<BE> as GLWESwitchingKeyEncryptSkDefault<BE>>::glwe_switching_key_encrypt_sk(
            module, res, sk_in, sk_out, enc_infos, source_xe, source_xa, scratch,
        )
    }

    fn glwe_switching_key_encrypt_pk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWESwitchingKeyEncryptPkDefault<BE>>::glwe_switching_key_encrypt_pk_tmp_bytes(module, infos)
    }

    fn glwe_tensor_key_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWETensorKeyEncryptSkDefault<BE>>::glwe_tensor_key_encrypt_sk_tmp_bytes(module, infos)
    }

    fn glwe_tensor_key_encrypt_sk_default<R, S, E>(
        module: &Module<BE>,
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
        <Module<BE> as GLWETensorKeyEncryptSkDefault<BE>>::glwe_tensor_key_encrypt_sk(
            module, res, sk, enc_infos, source_xe, source_xa, scratch,
        )
    }

    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWEToLWESwitchingKeyEncryptSkDefault<BE>>::glwe_to_lwe_key_encrypt_sk_tmp_bytes(module, infos)
    }

    fn glwe_to_lwe_key_encrypt_sk_default<R, S1, S2, E>(
        module: &Module<BE>,
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
        <Module<BE> as GLWEToLWESwitchingKeyEncryptSkDefault<BE>>::glwe_to_lwe_key_encrypt_sk(
            module, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch,
        )
    }

    fn lwe_switching_key_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as LWESwitchingKeyEncryptDefault<BE>>::lwe_switching_key_encrypt_sk_tmp_bytes(module, infos)
    }

    fn lwe_switching_key_encrypt_sk_default<R, S1, S2, E>(
        module: &Module<BE>,
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
        <Module<BE> as LWESwitchingKeyEncryptDefault<BE>>::lwe_switching_key_encrypt_sk(
            module, res, sk_lwe_in, sk_lwe_out, enc_infos, source_xe, source_xa, scratch,
        )
    }

    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as LWEToGLWESwitchingKeyEncryptSkDefault<BE>>::lwe_to_glwe_key_encrypt_sk_tmp_bytes(module, infos)
    }

    fn lwe_to_glwe_key_encrypt_sk_default<R, S1, S2, E>(
        module: &Module<BE>,
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
        <Module<BE> as LWEToGLWESwitchingKeyEncryptSkDefault<BE>>::lwe_to_glwe_key_encrypt_sk(
            module, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch,
        )
    }

    fn glwe_automorphism_key_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismKeyEncryptSkDefault<BE>>::glwe_automorphism_key_encrypt_sk_tmp_bytes(module, infos)
    }

    fn glwe_automorphism_key_encrypt_sk_default<R, S, E>(
        module: &Module<BE>,
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
        <Module<BE> as GLWEAutomorphismKeyEncryptSkDefault<BE>>::glwe_automorphism_key_encrypt_sk(
            module, res, p, sk, enc_infos, source_xe, source_xa, scratch,
        )
    }

    fn glwe_automorphism_key_encrypt_pk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismKeyEncryptPkDefault<BE>>::glwe_automorphism_key_encrypt_pk_tmp_bytes(module, infos)
    }

    fn glwe_compressed_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as GLWECompressedEncryptSkDefault<BE>>::glwe_compressed_encrypt_sk_tmp_bytes(module, infos)
    }

    fn glwe_compressed_encrypt_sk_default<R, P, S, E>(
        module: &Module<BE>,
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
        <Module<BE> as GLWECompressedEncryptSkDefault<BE>>::glwe_compressed_encrypt_sk(
            module, res, pt, sk, seed_xa, enc_infos, source_xe, scratch,
        )
    }

    fn gglwe_compressed_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GGLWECompressedEncryptSkDefault<BE>>::gglwe_compressed_encrypt_sk_tmp_bytes(module, infos)
    }

    fn gglwe_compressed_encrypt_sk_default<R, P, S, E>(
        module: &Module<BE>,
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
        <Module<BE> as GGLWECompressedEncryptSkDefault<BE>>::gglwe_compressed_encrypt_sk(
            module, res, pt, sk, seed, enc_infos, source_xe, scratch,
        )
    }

    fn ggsw_compressed_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        <Module<BE> as GGSWCompressedEncryptSkDefault<BE>>::ggsw_compressed_encrypt_sk_tmp_bytes(module, infos)
    }

    fn ggsw_compressed_encrypt_sk_default<R, P, S, E>(
        module: &Module<BE>,
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
        <Module<BE> as GGSWCompressedEncryptSkDefault<BE>>::ggsw_compressed_encrypt_sk(
            module, res, pt, sk, seed_xa, enc_infos, source_xe, scratch,
        )
    }

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GGLWEToGGSWKeyCompressedEncryptSkDefault<BE>>::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(module, infos)
    }

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_default<R, S, E>(
        module: &Module<BE>,
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
        <Module<BE> as GGLWEToGGSWKeyCompressedEncryptSkDefault<BE>>::gglwe_to_ggsw_key_encrypt_sk(
            module, res, sk, seed_xa, enc_infos, source_xe, scratch,
        )
    }

    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismKeyCompressedEncryptSkDefault<BE>>::glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(
            module, infos,
        )
    }

    fn glwe_automorphism_key_compressed_encrypt_sk_default<R, S, E>(
        module: &Module<BE>,
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
        <Module<BE> as GLWEAutomorphismKeyCompressedEncryptSkDefault<BE>>::glwe_automorphism_key_compressed_encrypt_sk(
            module, res, p, sk, seed_xa, enc_infos, source_xe, scratch,
        )
    }

    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWESwitchingKeyCompressedEncryptSkDefault<BE>>::glwe_switching_key_compressed_encrypt_sk_tmp_bytes(
            module, infos,
        )
    }

    fn glwe_switching_key_compressed_encrypt_sk_default<R, S1, S2, E>(
        module: &Module<BE>,
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
        <Module<BE> as GLWESwitchingKeyCompressedEncryptSkDefault<BE>>::glwe_switching_key_compressed_encrypt_sk(
            module, res, sk_in, sk_out, seed_xa, enc_infos, source_xe, scratch,
        )
    }

    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_default<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        <Module<BE> as GLWETensorKeyCompressedEncryptSkDefault<BE>>::glwe_tensor_key_compressed_encrypt_sk_tmp_bytes(
            module, infos,
        )
    }

    fn glwe_tensor_key_compressed_encrypt_sk_default<R, S, E>(
        module: &Module<BE>,
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
        <Module<BE> as GLWETensorKeyCompressedEncryptSkDefault<BE>>::glwe_tensor_key_compressed_encrypt_sk(
            module, res, sk, seed_xa, enc_infos, source_xe, scratch,
        )
    }
}

#[macro_export]
macro_rules! impl_core_encryption_default_methods {
    ($be:ty) => {
        fn lwe_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::LWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::lwe_encrypt_sk_tmp_bytes_default(module, infos)
        }

        fn lwe_encrypt_sk<R, P, S, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            pt: &P,
            sk: &S,
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            source_xa: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::LWEToMut,
            P: $crate::layouts::LWEPlaintextToRef,
            S: $crate::layouts::LWESecretToRef,
            E: $crate::EncryptionInfos,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::lwe_encrypt_sk_default(
                module, res, pt, sk, enc_infos, source_xe, source_xa, scratch,
            )
        }

        fn glwe_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_encrypt_sk_tmp_bytes_default(module, infos)
        }

        fn glwe_encrypt_sk<R, P, S, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            pt: &P,
            sk: &S,
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            source_xa: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut,
            P: $crate::layouts::GLWEPlaintextToRef,
            E: $crate::EncryptionInfos,
            S: $crate::layouts::GLWESecretPreparedToRef<$be>,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_encrypt_sk_default(
                module, res, pt, sk, enc_infos, source_xe, source_xa, scratch,
            )
        }

        fn glwe_encrypt_zero_sk<R, E, S>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            sk: &S,
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            source_xa: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut,
            E: $crate::EncryptionInfos,
            S: $crate::layouts::GLWESecretPreparedToRef<$be>,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_encrypt_zero_sk_default(
                module, res, sk, enc_infos, source_xe, source_xa, scratch,
            )
        }

        fn glwe_encrypt_pk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_encrypt_pk_tmp_bytes_default(module, infos)
        }

        fn glwe_encrypt_pk<R, P, K, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            pt: &P,
            pk: &K,
            enc_infos: &E,
            source_xu: &mut poulpy_hal::source::Source,
            source_xe: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            P: $crate::layouts::GLWEPlaintextToRef + $crate::layouts::GLWEInfos,
            E: $crate::EncryptionInfos,
            K: $crate::layouts::GLWEPreparedToRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_encrypt_pk_default(
                module, res, pt, pk, enc_infos, source_xu, source_xe, scratch,
            )
        }

        fn glwe_encrypt_zero_pk<R, K, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            pk: &K,
            enc_infos: &E,
            source_xu: &mut poulpy_hal::source::Source,
            source_xe: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            E: $crate::EncryptionInfos,
            K: $crate::layouts::GLWEPreparedToRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_encrypt_zero_pk_default(
                module, res, pk, enc_infos, source_xu, source_xe, scratch,
            )
        }

        fn glwe_public_key_generate<R, S, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            sk: &S,
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            source_xa: &mut poulpy_hal::source::Source,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::GetDistributionMut + $crate::layouts::GLWEInfos,
            E: $crate::EncryptionInfos,
            S: $crate::layouts::GLWESecretPreparedToRef<$be> + $crate::GetDistribution,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_public_key_generate_default(
                module, res, sk, enc_infos, source_xe, source_xa,
            )
        }

        fn gglwe_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::gglwe_encrypt_sk_tmp_bytes_default(module, infos)
        }

        fn gglwe_encrypt_sk<R, P, S, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            pt: &P,
            sk: &S,
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            source_xa: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWEToMut,
            P: poulpy_hal::layouts::ScalarZnxToRef,
            E: $crate::EncryptionInfos,
            S: $crate::layouts::GLWESecretPreparedToRef<$be>,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::gglwe_encrypt_sk_default(
                module, res, pt, sk, enc_infos, source_xe, source_xa, scratch,
            )
        }

        fn ggsw_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGSWInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::ggsw_encrypt_sk_tmp_bytes_default(module, infos)
        }

        fn ggsw_encrypt_sk<R, P, S, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            pt: &P,
            sk: &S,
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            source_xa: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGSWToMut,
            P: poulpy_hal::layouts::ScalarZnxToRef,
            E: $crate::EncryptionInfos,
            S: $crate::layouts::GLWESecretPreparedToRef<$be>,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::ggsw_encrypt_sk_default(
                module, res, pt, sk, enc_infos, source_xe, source_xa, scratch,
            )
        }

        fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes_default(module, infos)
        }

        fn gglwe_to_ggsw_key_encrypt_sk<R, S, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            sk: &S,
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            source_xa: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWEToGGSWKeyToMut,
            E: $crate::EncryptionInfos,
            S: $crate::layouts::GLWESecretToRef + $crate::GetDistribution + $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::gglwe_to_ggsw_key_encrypt_sk_default(
                module, res, sk, enc_infos, source_xe, source_xa, scratch,
            )
        }

        fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_switching_key_encrypt_sk_tmp_bytes_default(module, infos)
        }

        fn glwe_switching_key_encrypt_sk<R, S1, S2, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            sk_in: &S1,
            sk_out: &S2,
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            source_xa: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWEToMut + $crate::layouts::GLWESwitchingKeyDegreesMut + $crate::layouts::GGLWEInfos,
            E: $crate::EncryptionInfos,
            S1: $crate::layouts::GLWESecretToRef,
            S2: $crate::layouts::GLWESecretToRef,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_switching_key_encrypt_sk_default(
                module, res, sk_in, sk_out, enc_infos, source_xe, source_xa, scratch,
            )
        }

        fn glwe_switching_key_encrypt_pk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_switching_key_encrypt_pk_tmp_bytes_default(module, infos)
        }

        fn glwe_tensor_key_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_tensor_key_encrypt_sk_tmp_bytes_default(module, infos)
        }

        fn glwe_tensor_key_encrypt_sk<R, S, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            sk: &S,
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            source_xa: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWEToMut + $crate::layouts::GGLWEInfos,
            E: $crate::EncryptionInfos,
            S: $crate::layouts::GLWESecretToRef + $crate::GetDistribution + $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_tensor_key_encrypt_sk_default(
                module, res, sk, enc_infos, source_xe, source_xa, scratch,
            )
        }

        fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_to_lwe_key_encrypt_sk_tmp_bytes_default(module, infos)
        }

        fn glwe_to_lwe_key_encrypt_sk<R, S1, S2, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            sk_lwe: &S1,
            sk_glwe: &S2,
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            source_xa: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            S1: $crate::layouts::LWESecretToRef,
            S2: $crate::layouts::GLWESecretToRef,
            E: $crate::EncryptionInfos,
            R: $crate::layouts::GGLWEToMut + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_to_lwe_key_encrypt_sk_default(
                module, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch,
            )
        }

        fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::lwe_switching_key_encrypt_sk_tmp_bytes_default(module, infos)
        }

        fn lwe_switching_key_encrypt_sk<R, S1, S2, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            sk_lwe_in: &S1,
            sk_lwe_out: &S2,
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            source_xa: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWEToMut + $crate::layouts::GLWESwitchingKeyDegreesMut + $crate::layouts::GGLWEInfos,
            E: $crate::EncryptionInfos,
            S1: $crate::layouts::LWESecretToRef,
            S2: $crate::layouts::LWESecretToRef,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::lwe_switching_key_encrypt_sk_default(
                module, res, sk_lwe_in, sk_lwe_out, enc_infos, source_xe, source_xa, scratch,
            )
        }

        fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::lwe_to_glwe_key_encrypt_sk_tmp_bytes_default(module, infos)
        }

        fn lwe_to_glwe_key_encrypt_sk<R, S1, S2, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            sk_lwe: &S1,
            sk_glwe: &S2,
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            source_xa: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            S1: $crate::layouts::LWESecretToRef,
            S2: $crate::layouts::GLWESecretPreparedToRef<$be>,
            E: $crate::EncryptionInfos,
            R: $crate::layouts::GGLWEToMut + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::lwe_to_glwe_key_encrypt_sk_default(
                module, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch,
            )
        }

        fn glwe_automorphism_key_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_automorphism_key_encrypt_sk_tmp_bytes_default(module, infos)
        }

        fn glwe_automorphism_key_encrypt_sk<R, S, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            p: i64,
            sk: &S,
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            source_xa: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWEToMut + $crate::layouts::SetGaloisElement + $crate::layouts::GGLWEInfos,
            E: $crate::EncryptionInfos,
            S: $crate::layouts::GLWESecretToRef,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_automorphism_key_encrypt_sk_default(
                module, res, p, sk, enc_infos, source_xe, source_xa, scratch,
            )
        }

        fn glwe_automorphism_key_encrypt_pk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_automorphism_key_encrypt_pk_tmp_bytes_default(module, infos)
        }

        fn glwe_compressed_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_compressed_encrypt_sk_tmp_bytes_default(module, infos)
        }

        fn glwe_compressed_encrypt_sk<R, P, S, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            pt: &P,
            sk: &S,
            seed_xa: [u8; 32],
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWECompressedToMut + $crate::layouts::GLWECompressedSeedMut,
            P: $crate::layouts::GLWEPlaintextToRef,
            E: $crate::EncryptionInfos,
            S: $crate::layouts::GLWESecretPreparedToRef<$be>,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_compressed_encrypt_sk_default(
                module, res, pt, sk, seed_xa, enc_infos, source_xe, scratch,
            )
        }

        fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::gglwe_compressed_encrypt_sk_tmp_bytes_default(module, infos)
        }

        fn gglwe_compressed_encrypt_sk<R, P, S, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            pt: &P,
            sk: &S,
            seed: [u8; 32],
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWECompressedToMut + $crate::layouts::GGLWECompressedSeedMut,
            P: poulpy_hal::layouts::ScalarZnxToRef,
            E: $crate::EncryptionInfos,
            S: $crate::layouts::GLWESecretPreparedToRef<$be>,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::gglwe_compressed_encrypt_sk_default(
                module, res, pt, sk, seed, enc_infos, source_xe, scratch,
            )
        }

        fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGSWInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::ggsw_compressed_encrypt_sk_tmp_bytes_default(module, infos)
        }

        fn ggsw_compressed_encrypt_sk<R, P, S, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            pt: &P,
            sk: &S,
            seed_xa: [u8; 32],
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGSWCompressedToMut + $crate::layouts::GGSWCompressedSeedMut + $crate::layouts::GGSWInfos,
            P: poulpy_hal::layouts::ScalarZnxToRef,
            E: $crate::EncryptionInfos,
            S: $crate::layouts::GLWESecretPreparedToRef<$be>,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::ggsw_compressed_encrypt_sk_default(
                module, res, pt, sk, seed_xa, enc_infos, source_xe, scratch,
            )
        }

        fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes_default(
                module, infos,
            )
        }

        fn gglwe_to_ggsw_key_compressed_encrypt_sk<R, S, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            sk: &S,
            seed_xa: [u8; 32],
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWEToGGSWKeyCompressedToMut + $crate::layouts::GGLWEInfos,
            E: $crate::EncryptionInfos,
            S: $crate::layouts::GLWESecretToRef + $crate::GetDistribution + $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::gglwe_to_ggsw_key_compressed_encrypt_sk_default(
                module, res, sk, seed_xa, enc_infos, source_xe, scratch,
            )
        }

        fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes_default(
                module, infos,
            )
        }

        fn glwe_automorphism_key_compressed_encrypt_sk<R, S, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            p: i64,
            sk: &S,
            seed_xa: [u8; 32],
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWECompressedToMut
                + $crate::layouts::GGLWECompressedSeedMut
                + $crate::layouts::SetGaloisElement
                + $crate::layouts::GGLWEInfos,
            E: $crate::EncryptionInfos,
            S: $crate::layouts::GLWESecretToRef + $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_automorphism_key_compressed_encrypt_sk_default(
                module, res, p, sk, seed_xa, enc_infos, source_xe, scratch,
            )
        }

        fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_switching_key_compressed_encrypt_sk_tmp_bytes_default(
                module, infos,
            )
        }

        fn glwe_switching_key_compressed_encrypt_sk<R, S1, S2, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            sk_in: &S1,
            sk_out: &S2,
            seed_xa: [u8; 32],
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWECompressedToMut
                + $crate::layouts::GGLWECompressedSeedMut
                + $crate::layouts::GLWESwitchingKeyDegreesMut
                + $crate::layouts::GGLWEInfos,
            E: $crate::EncryptionInfos,
            S1: $crate::layouts::GLWESecretToRef,
            S2: $crate::layouts::GLWESecretToRef,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_switching_key_compressed_encrypt_sk_default(
                module, res, sk_in, sk_out, seed_xa, enc_infos, source_xe, scratch,
            )
        }

        fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes<A>(module: &poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
        where
            A: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_default(
                module, infos,
            )
        }

        fn glwe_tensor_key_compressed_encrypt_sk<R, S, E>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            sk: &S,
            seed_xa: [u8; 32],
            enc_infos: &E,
            source_xe: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWECompressedToMut + $crate::layouts::GGLWEInfos + $crate::layouts::GGLWECompressedSeedMut,
            E: $crate::EncryptionInfos,
            S: $crate::layouts::GLWESecretToRef + $crate::GetDistribution + $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreEncryptionDefaults<$be>>::glwe_tensor_key_compressed_encrypt_sk_default(
                module, res, sk, seed_xa, enc_infos, source_xe, scratch,
            )
        }
    };
}
