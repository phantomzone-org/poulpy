use poulpy_hal::{
    layouts::{Backend, DataMut, Scratch},
    source::Source,
};

use poulpy_core::{
    GetDistribution, ScratchTakeCore,
    layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToRef, LWEInfos, LWESecretToRef},
};

use crate::bin_fhe::blind_rotation::{BlindRotationAlgo, BlindRotationKey};

pub trait BlindRotationKeyEncryptSk<BRA: BlindRotationAlgo, B: Backend> {
    fn blind_rotation_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    #[allow(clippy::too_many_arguments)]
    fn blind_rotation_key_encrypt_sk<D, S0, S1>(
        &self,
        res: &mut BlindRotationKey<D, BRA>,
        sk_glwe: &S0,
        sk_lwe: &S1,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        D: DataMut,
        S0: GLWESecretPreparedToRef<B> + GLWEInfos,
        S1: LWESecretToRef + LWEInfos + GetDistribution;
}

impl<D: DataMut, BRA: BlindRotationAlgo> BlindRotationKey<D, BRA> {
    pub fn encrypt_sk<M, S0, S1, BE: Backend>(
        &mut self,
        module: &M,
        sk_glwe: &S0,
        sk_lwe: &S1,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S0: GLWESecretPreparedToRef<BE> + GLWEInfos,
        S1: LWESecretToRef + LWEInfos + GetDistribution,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: BlindRotationKeyEncryptSk<BRA, BE>,
    {
        module.blind_rotation_key_encrypt_sk(self, sk_glwe, sk_lwe, source_xa, source_xe, scratch);
    }
}

impl<BRA: BlindRotationAlgo> BlindRotationKey<Vec<u8>, BRA> {
    pub fn encrypt_sk_tmp_bytes<A, M, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGSWInfos,
        M: BlindRotationKeyEncryptSk<BRA, BE>,
    {
        module.blind_rotation_key_encrypt_sk_tmp_bytes(infos)
    }
}
