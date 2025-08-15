use backend::hal::{
    api::{
        ScratchAvailable, TakeScalarZnx, TakeVecZnx, TakeVecZnxDft, VecZnxAddScalarInplace, VecZnxAutomorphism,
        VecZnxSwithcDegree,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};
use sampling::source::Source;

use crate::{
    TakeGLWESecret, TakeGLWESecretPrepared,
    layouts::{
        GLWESecret,
        compressed::{GGLWEAutomorphismKeyCompressed, GGLWESwitchingKeyCompressed},
    },
};

use crate::trait_families::{GGLWEAutomorphismKeyEncryptSkFamily, GLWESecretPreparedModuleFamily};

impl GGLWEAutomorphismKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: GGLWEAutomorphismKeyEncryptSkFamily<B> + GLWESecretPreparedModuleFamily<B>,
    {
        GGLWESwitchingKeyCompressed::encrypt_sk_scratch_space(module, n, basek, k, rank, rank) + GLWESecret::bytes_of(n, rank)
    }
}

impl<DataSelf: DataMut> GGLWEAutomorphismKeyCompressed<DataSelf> {
    pub fn encrypt_sk<DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        p: i64,
        sk: &GLWESecret<DataSk>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GGLWEAutomorphismKeyEncryptSkFamily<B>
            + VecZnxSwithcDegree
            + VecZnxAutomorphism
            + VecZnxAddScalarInplace
            + GLWESecretPreparedModuleFamily<B>,
        Scratch<B>: ScratchAvailable + TakeScalarZnx + TakeVecZnxDft<B> + TakeGLWESecretPrepared<B> + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            use crate::layouts::Infos;

            assert_eq!(self.n(), sk.n());
            assert_eq!(self.rank_out(), self.rank_in());
            assert_eq!(sk.rank(), self.rank());
            assert!(
                scratch.available()
                    >= GGLWEAutomorphismKeyCompressed::encrypt_sk_scratch_space(
                        module,
                        sk.n(),
                        self.basek(),
                        self.k(),
                        self.rank()
                    ),
                "scratch.available(): {} < AutomorphismKey::encrypt_sk_scratch_space(module, self.rank()={}, self.size()={}): {}",
                scratch.available(),
                self.rank(),
                self.size(),
                GGLWEAutomorphismKeyCompressed::encrypt_sk_scratch_space(module, sk.n(), self.basek(), self.k(), self.rank())
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
            .encrypt_sk(module, &sk, &sk_out, seed_xa, source_xe, sigma, scratch_1);

        self.p = p;
    }
}
