use poulpy_hal::{
    api::{
        DFT, IDFTConsume, ScratchAvailable, SvpApplyInplace, SvpPPolAllocBytes, SvpPrepare, TakeScalarZnx, TakeVecZnx,
        TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxBigNormalize,
        VecZnxDftAllocBytes, VecZnxFillUniform, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub,
        VecZnxSubABInplace, VecZnxSwithcDegree,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use crate::{
    TakeGLWESecret, TakeGLWESecretPrepared,
    layouts::{
        GLWESecret,
        compressed::{GGLWEAutomorphismKeyCompressed, GGLWESwitchingKeyCompressed},
    },
};

impl GGLWEAutomorphismKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes + SvpPPolAllocBytes,
    {
        GGLWESwitchingKeyCompressed::encrypt_sk_scratch_space(module, basek, k, rank, rank)
            + GLWESecret::bytes_of(module.n(), rank)
    }
}

impl<DataSelf: DataMut> GGLWEAutomorphismKeyCompressed<DataSelf> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        p: i64,
        sk: &GLWESecret<DataSk>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxAutomorphism
            + SvpPrepare<B>
            + SvpPPolAllocBytes
            + VecZnxSwithcDegree
            + VecZnxDftAllocBytes
            + VecZnxBigNormalize<B>
            + DFT<B>
            + SvpApplyInplace<B>
            + IDFTConsume<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxFillUniform
            + VecZnxSubABInplace
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxAddNormal
            + VecZnxNormalize<B>
            + VecZnxSub
            + VecZnxAddScalarInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx + TakeScalarZnx + TakeGLWESecretPrepared<B>,
    {
        #[cfg(debug_assertions)]
        {
            use crate::layouts::Infos;

            assert_eq!(self.n(), sk.n());
            assert_eq!(self.rank_out(), self.rank_in());
            assert_eq!(sk.rank(), self.rank());
            assert!(
                scratch.available()
                    >= GGLWEAutomorphismKeyCompressed::encrypt_sk_scratch_space(module, self.basek(), self.k(), self.rank()),
                "scratch.available(): {} < AutomorphismKey::encrypt_sk_scratch_space(module, self.rank()={}, self.size()={}): {}",
                scratch.available(),
                self.rank(),
                self.size(),
                GGLWEAutomorphismKeyCompressed::encrypt_sk_scratch_space(module, self.basek(), self.k(), self.rank())
            )
        }

        let (mut sk_out, scratch_1) = scratch.take_glwe_secret(sk.n(), sk.rank());

        {
            (0..self.rank()).for_each(|i| {
                module.vec_znx_automorphism(
                    module.galois_element_inv(p),
                    &mut sk_out.data.as_vec_znx_mut(),
                    i,
                    &sk.data.as_vec_znx(),
                    i,
                );
            });
        }

        self.key
            .encrypt_sk(module, sk, &sk_out, seed_xa, source_xe, scratch_1);

        self.p = p;
    }
}
