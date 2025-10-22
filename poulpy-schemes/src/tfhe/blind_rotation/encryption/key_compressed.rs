use poulpy_core::{
    GetDistribution,
    layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToRef, LWEInfos, LWESecretToRef},
};
use poulpy_hal::{
    layouts::{Backend, DataMut, Scratch},
    source::Source,
};

use crate::tfhe::blind_rotation::{BlindRotationAlgo, BlindRotationKeyCompressed};

pub trait BlindRotationKeyCompressedEncryptSk<B: Backend, BRA: BlindRotationAlgo> {
    fn blind_rotation_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    #[allow(clippy::too_many_arguments)]
    fn blind_rotation_key_compressed_encrypt_sk<D, S0, S1>(
        &self,
        res: &mut BlindRotationKeyCompressed<D, BRA>,
        sk_glwe: &S0,
        sk_lwe: &S1,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        D: DataMut,
        S0: GLWESecretPreparedToRef<B> + GLWEInfos,
        S1: LWESecretToRef + LWEInfos + GetDistribution;
}
