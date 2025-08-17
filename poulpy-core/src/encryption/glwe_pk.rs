use poulpy_backend::hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyInplace, VecZnxAddInplace, VecZnxAddNormal, VecZnxBigNormalize,
        VecZnxDftAllocBytes, VecZnxDftFromVecZnx, VecZnxDftToVecZnxBigConsume, VecZnxFillUniform, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace,
    },
    layouts::{Backend, DataMut, DataRef, Module, ScratchOwned},
    oep::{ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeVecZnxDftImpl, TakeVecZnxImpl},
    source::Source,
};

use crate::layouts::{GLWECiphertext, GLWEPublicKey, Infos, prepared::GLWESecretPrepared};

impl<D: DataMut> GLWEPublicKey<D> {
    pub fn generate_from_sk<S: DataRef, B>(
        &mut self,
        module: &Module<B>,
        sk: &GLWESecretPrepared<S, B>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
    ) where
        Module<B>:,
        Module<B>: VecZnxDftAllocBytes
            + VecZnxBigNormalize<B>
            + VecZnxDftFromVecZnx<B>
            + SvpApplyInplace<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxFillUniform
            + VecZnxSubABInplace
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxAddNormal
            + VecZnxNormalize<B>
            + VecZnxSub,
        B: Backend
            + ScratchOwnedAllocImpl<B>
            + ScratchOwnedBorrowImpl<B>
            + TakeVecZnxDftImpl<B>
            + ScratchAvailableImpl<B>
            + TakeVecZnxImpl<B>,
    {
        #[cfg(debug_assertions)]
        {
            use crate::Distribution;

            assert_eq!(self.n(), sk.n());

            if sk.dist == Distribution::NONE {
                panic!("invalid sk: SecretDistribution::NONE")
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
