use backend::hal::{
    api::{
        ScratchAvailable, TakeScalarZnx, TakeVecZnx, TakeVecZnxDft, VecZnxAddScalarInplace, VecZnxAutomorphismInplace,
        VecZnxSwithcDegree, ZnxView, ZnxViewMut,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};
use sampling::source::Source;

use crate::{
    TakeGLWESecret, TakeGLWESecretPrepared,
    layouts::{GGLWESwitchingKey, GLWESecret, Infos, LWESecret, LWESwitchingKey, prepared::GLWESecretPrepared},
};

use crate::trait_families::{GGLWEEncryptSkFamily, GGLWESwitchingKeyEncryptSkFamily, GLWESecretPreparedModuleFamily};

impl LWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B> + GLWESecretPreparedModuleFamily<B>,
    {
        GLWESecret::bytes_of(n, 1)
            + GLWESecretPrepared::bytes_of(module, n, 1)
            + GGLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k, 1, 1)
    }
}

impl<D: DataMut> LWESwitchingKey<D> {
    pub fn encrypt_sk<DIn, DOut, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_lwe_in: &LWESecret<DIn>,
        sk_lwe_out: &LWESecret<DOut>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        DIn: DataRef,
        DOut: DataRef,
        Module<B>: GGLWESwitchingKeyEncryptSkFamily<B>
            + VecZnxAutomorphismInplace
            + VecZnxSwithcDegree
            + VecZnxAddScalarInplace
            + GLWESecretPreparedModuleFamily<B>,
        Scratch<B>: ScratchAvailable + TakeScalarZnx + TakeVecZnxDft<B> + TakeGLWESecretPrepared<B> + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe_in.n() <= self.n());
            assert!(sk_lwe_out.n() <= self.n());
            assert!(self.n() <= module.n());
        }

        let (mut sk_in_glwe, scratch1) = scratch.take_glwe_secret(self.n(), 1);
        let (mut sk_out_glwe, scratch2) = scratch1.take_glwe_secret(self.n(), 1);

        sk_out_glwe.data.at_mut(0, 0)[..sk_lwe_out.n()].copy_from_slice(sk_lwe_out.data.at(0, 0));
        sk_out_glwe.data.at_mut(0, 0)[sk_lwe_out.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_out_glwe.data.as_vec_znx_mut(), 0);

        sk_in_glwe.data.at_mut(0, 0)[..sk_lwe_in.n()].copy_from_slice(sk_lwe_in.data.at(0, 0));
        sk_in_glwe.data.at_mut(0, 0)[sk_lwe_in.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_in_glwe.data.as_vec_znx_mut(), 0);

        self.0.encrypt_sk(
            module,
            &sk_in_glwe,
            &sk_out_glwe,
            source_xa,
            source_xe,
            sigma,
            scratch2,
        );
    }
}
