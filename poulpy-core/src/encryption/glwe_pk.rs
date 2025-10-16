use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftBytesOf, VecZnxNormalizeTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, ScratchOwned},
    source::Source,
};

use crate::{
    encryption::glwe_ct::GLWEEncryptZeroSk,
    layouts::{
        GLWE, GLWEPublicKey, GLWEPublicKeyToMut,
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

pub trait GLWEPublicKeyGenerate<B: Backend> {
    fn glwe_public_key_generate<R, S>(&self, res: &mut R, sk: &S, source_xa: &mut Source, source_xe: &mut Source)
    where
        R: GLWEPublicKeyToMut,
        S: GLWESecretPreparedToRef<B>;
}

impl<B: Backend> GLWEPublicKeyGenerate<B> for Module<B>
where
    Module<B>: GLWEEncryptZeroSk<B> + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    fn glwe_public_key_generate<R, S>(&self, res: &mut R, sk: &S, source_xa: &mut Source, source_xe: &mut Source)
    where
        R: GLWEPublicKeyToMut,
        S: GLWESecretPreparedToRef<B>,
    {
        let res: &mut GLWEPublicKey<&mut [u8]> = &mut res.to_mut();
        let sk: &GLWESecretPrepared<&[u8], B> = &sk.to_ref();

        #[cfg(debug_assertions)]
        {
            use crate::{Distribution, layouts::LWEInfos};

            assert_eq!(res.n(), self.n() as u32);
            assert_eq!(sk.n(), self.n() as u32);

            if sk.dist == Distribution::NONE {
                panic!("invalid sk: SecretDistribution::NONE")
            }
        }

        // Its ok to allocate scratch space here since pk is usually generated only once.
        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GLWE::encrypt_sk_tmp_bytes(self, res));

        let mut tmp: GLWE<Vec<u8>> = GLWE::alloc_from_infos(self, res);

        tmp.encrypt_zero_sk(self, sk, source_xa, source_xe, scratch.borrow());
        res.dist = sk.dist;
    }
}

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
