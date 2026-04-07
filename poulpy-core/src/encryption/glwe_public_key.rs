use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, DataMut, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    Distribution, EncryptionInfos, GLWEEncryptSk, GetDistribution, GetDistributionMut, ScratchTakeCore,
    layouts::{
        GLWEInfos, GLWEPublicKey, GLWEToMut,
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

impl<D: DataMut> GLWEPublicKey<D> {
    pub fn generate<S, M, E, BE: Backend>(
        &mut self,
        module: &M,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
    ) where
        S: GLWESecretPreparedToRef<BE> + GetDistribution,
        E: EncryptionInfos,
        M: GLWEPublicKeyGenerate<BE>,
    {
        module.glwe_public_key_generate(self, sk, enc_infos, source_xe, source_xa);
    }
}

pub trait GLWEPublicKeyGenerate<BE: Backend> {
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
        S: GLWESecretPreparedToRef<BE> + GetDistribution;
}

impl<BE: Backend> GLWEPublicKeyGenerate<BE> for Module<BE>
where
    Self: GLWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
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
        {
            let sk: &GLWESecretPrepared<&[u8], BE> = &sk.to_ref();

            assert_eq!(res.n(), self.n() as u32);
            assert_eq!(sk.n(), self.n() as u32);

            if sk.dist == Distribution::NONE {
                panic!("invalid sk: SecretDistribution::NONE")
            }

            // Its ok to allocate scratch space here since pk is usually generated only once.
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.glwe_encrypt_sk_tmp_bytes(res));
            res.to_mut()
                .encrypt_zero_sk(self, sk, enc_infos, source_xe, source_xa, scratch.borrow());
        }
        *res.dist_mut() = *sk.dist();
    }
}
