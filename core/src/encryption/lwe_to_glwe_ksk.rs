use backend::hal::{
    api::{
        ScratchAvailable, TakeScalarZnx, TakeVecZnx, TakeVecZnxDft, VecZnxAddScalarInplace, VecZnxAutomorphismInplace,
        VecZnxSwithcDegree, ZnxView, ZnxViewMut,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};
use sampling::source::Source;

use crate::{
    TakeGLWESecret, TakeGLWESecretExec,
    layouts::{GGLWESwitchingKey, GLWESecret, LWESecret, LWEToGLWESwitchingKey},
};

use crate::trait_families::{GGLWEEncryptSkFamily, GGLWESwitchingKeyEncryptSkFamily, GLWESecretExecModuleFamily};

impl LWEToGLWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize, rank_out: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B> + GLWESecretExecModuleFamily<B>,
    {
        GGLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k, 1, rank_out) + GLWESecret::bytes_of(n, 1)
    }
}

impl<D: DataMut> LWEToGLWESwitchingKey<D> {
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
            + GLWESecretExecModuleFamily<B>,
        Scratch<B>: ScratchAvailable + TakeScalarZnx + TakeVecZnxDft<B> + TakeGLWESecretExec<B> + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe.n() <= module.n());
        }

        let (mut sk_lwe_as_glwe, scratch1) = scratch.take_glwe_secret(sk_glwe.n(), 1);
        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n()].copy_from_slice(sk_lwe.data.at(0, 0));
        sk_lwe_as_glwe.data.at_mut(0, 0)[sk_lwe.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0);

        self.0.encrypt_sk(
            module,
            &sk_lwe_as_glwe,
            &sk_glwe,
            source_xa,
            source_xe,
            sigma,
            scratch1,
        );
    }
}
