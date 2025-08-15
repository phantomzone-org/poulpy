use backend::hal::{
    api::{
        ScratchAvailable, TakeScalarZnx, TakeVecZnx, TakeVecZnxDft, VecZnxAddScalarInplace, VecZnxAutomorphismInplace,
        VecZnxSwithcDegree, ZnxView, ZnxViewMut, ZnxZero,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};
use sampling::source::Source;

use crate::{
    TakeGLWESecret, TakeGLWESecretPrepared,
    layouts::{GGLWESwitchingKey, GLWESecret, GLWEToLWESwitchingKey, LWESecret, prepared::GLWESecretPrepared},
};

use crate::trait_families::{GGLWEEncryptSkFamily, GGLWESwitchingKeyEncryptSkFamily, GLWESecretPreparedModuleFamily};

impl GLWEToLWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize, rank_in: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B> + GLWESecretPreparedModuleFamily<B>,
    {
        GLWESecretPrepared::bytes_of(module, n, rank_in)
            + (GGLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k, rank_in, 1) | GLWESecret::bytes_of(n, rank_in))
    }
}

impl<D: DataMut> GLWEToLWESwitchingKey<D> {
    pub fn encrypt_sk<DLwe, DGlwe, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        DLwe: DataRef,
        DGlwe: DataRef,
        Module<B>: GGLWESwitchingKeyEncryptSkFamily<B>
            + VecZnxAutomorphismInplace
            + VecZnxSwithcDegree
            + VecZnxAddScalarInplace
            + GLWESecretPreparedModuleFamily<B>,
        Scratch<B>: ScratchAvailable + TakeScalarZnx + TakeVecZnxDft<B> + TakeGLWESecretPrepared<B> + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe.n() <= module.n());
        }

        let (mut sk_lwe_as_glwe, scratch1) = scratch.take_glwe_secret(sk_glwe.n(), 1);
        sk_lwe_as_glwe.data.zero();
        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n()].copy_from_slice(sk_lwe.data.at(0, 0));
        module.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0);

        self.0.encrypt_sk(
            module,
            sk_glwe,
            &sk_lwe_as_glwe,
            source_xa,
            source_xe,
            sigma,
            scratch1,
        );
    }
}
