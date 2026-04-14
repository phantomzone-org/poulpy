use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    Distribution, EncryptionInfos, GLWEEncryptSk, GetDistribution, GetDistributionMut, ScratchTakeCore,
    layouts::{
        GLWEInfos, GLWEToMut,
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

#[doc(hidden)]
pub trait GLWEPublicKeyGenerateDefault<BE: Backend> {
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

impl<BE: Backend> GLWEPublicKeyGenerateDefault<BE> for Module<BE>
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
            let mut res_glwe = res.to_mut();
            self.glwe_encrypt_zero_sk(&mut res_glwe, sk, enc_infos, source_xe, source_xa, scratch.borrow());
        }
        *res.dist_mut() = *sk.dist();
    }
}
