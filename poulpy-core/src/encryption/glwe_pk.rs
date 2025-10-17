use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    Distribution, ScratchTakeCore,
    encryption::glwe_ct::GLWEEncryptSk,
    layouts::{
        GLWE, GLWEPublicKey, GLWEPublicKeyToMut, LWEInfos,
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

impl<D: DataMut> GLWEPublicKey<D> {
    pub fn generate<S: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk: &GLWESecretPrepared<S, B>,
        source_xa: &mut Source,
        source_xe: &mut Source,
    ) where
        Module<B>: GLWEPublicKeyGenerate<B>,
    {
        module.glwe_public_key_generate(self, sk, source_xa, source_xe);
    }
}

pub trait GLWEPublicKeyGenerate<B: Backend> {
    fn glwe_public_key_generate<R, S>(&self, res: &mut R, sk: &S, source_xa: &mut Source, source_xe: &mut Source)
    where
        R: GLWEPublicKeyToMut,
        S: GLWESecretPreparedToRef<B>;
}

impl<BE: Backend> GLWEPublicKeyGenerate<BE> for Module<BE>
where
    Module<BE>: GLWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_public_key_generate<R, S>(&self, res: &mut R, sk: &S, source_xa: &mut Source, source_xe: &mut Source)
    where
        R: GLWEPublicKeyToMut,
        S: GLWESecretPreparedToRef<BE>,
    {
        let res: &mut GLWEPublicKey<&mut [u8]> = &mut res.to_mut();
        let sk: &GLWESecretPrepared<&[u8], BE> = &sk.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk.n(), self.n() as u32);

        if sk.dist == Distribution::NONE {
            panic!("invalid sk: SecretDistribution::NONE")
        }

        // Its ok to allocate scratch space here since pk is usually generated only once.
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.glwe_encrypt_sk_tmp_bytes(res));

        let mut tmp: GLWE<Vec<u8>> = GLWE::alloc_from_infos(self, res);

        tmp.encrypt_zero_sk(self, sk, source_xa, source_xe, scratch.borrow());
        res.dist = sk.dist;
    }
}
