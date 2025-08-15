use backend::hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, DataMut, DataRef, Module, ScratchOwned},
    oep::{ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeVecZnxDftImpl, TakeVecZnxImpl},
};
use sampling::source::Source;

use crate::layouts::{GLWECiphertext, GLWEPublicKey, Infos, prepared::GLWESecretPrepared};

use crate::trait_families::GLWEEncryptSkFamily;

impl<D: DataMut> GLWEPublicKey<D> {
    pub fn generate_from_sk<S: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk: &GLWESecretPrepared<S, B>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
    ) where
        Module<B>: GLWEEncryptSkFamily<B>,
        B: ScratchOwnedAllocImpl<B>
            + ScratchOwnedBorrowImpl<B>
            + TakeVecZnxDftImpl<B>
            + ScratchAvailableImpl<B>
            + TakeVecZnxImpl<B>,
    {
        #[cfg(debug_assertions)]
        {
            use crate::Distribution;

            assert_eq!(self.n(), sk.n());

            match sk.dist {
                Distribution::NONE => panic!("invalid sk: SecretDistribution::NONE"),
                _ => {}
            }
        }

        // Its ok to allocate scratch space here since pk is usually generated only once.
        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GLWECiphertext::encrypt_sk_scratch_space(
            module,
            self.n(),
            self.basek(),
            self.k(),
        ));

        let mut tmp: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(self.n(), self.basek(), self.k(), self.rank());
        tmp.encrypt_zero_sk(module, sk, source_xa, source_xe, sigma, scratch.borrow());
        self.dist = sk.dist;
    }
}
