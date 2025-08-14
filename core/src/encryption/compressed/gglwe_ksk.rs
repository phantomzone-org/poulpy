use backend::hal::{
    api::{ScratchAvailable, TakeScalarZnx, TakeVecZnx, TakeVecZnxDft, VecZnxAddScalarInplace, VecZnxSwithcDegree},
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, Scratch},
};
use sampling::source::Source;

use crate::{
    TakeGLWESecretExec,
    layouts::{
        GGLWECiphertext, GLWESecret, GGLWESwitchingKey, Infos, compressed::GGLWESwitchingKeyCompressed, prepared::GLWESecretExec,
    },
};

use crate::trait_families::{GGLWESwitchingKeyEncryptSkFamily, GLWESecretExecModuleFamily};

impl GGLWESwitchingKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize
    where
        Module<B>: GGLWESwitchingKeyEncryptSkFamily<B> + GLWESecretExecModuleFamily<B>,
    {
        (GGLWECiphertext::encrypt_sk_scratch_space(module, n, basek, k) | ScalarZnx::alloc_bytes(n, 1))
            + ScalarZnx::alloc_bytes(n, rank_in)
            + GLWESecretExec::bytes_of(module, n, rank_out)
    }
}

impl<DataSelf: DataMut> GGLWESwitchingKeyCompressed<DataSelf> {
    pub fn encrypt_sk<DataSkIn: DataRef, DataSkOut: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_in: &GLWESecret<DataSkIn>,
        sk_out: &GLWESecret<DataSkOut>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>:
            GGLWESwitchingKeyEncryptSkFamily<B> + VecZnxSwithcDegree + VecZnxAddScalarInplace + GLWESecretExecModuleFamily<B>,
        Scratch<B>: ScratchAvailable + TakeScalarZnx + TakeVecZnxDft<B> + TakeGLWESecretExec<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_in.n() <= module.n());
            assert!(sk_out.n() <= module.n());
            assert!(
                scratch.available()
                    >= GGLWESwitchingKey::encrypt_sk_scratch_space(
                        module,
                        sk_out.n(),
                        self.basek(),
                        self.k(),
                        self.rank_in(),
                        self.rank_out()
                    ),
                "scratch.available()={} < GLWESwitchingKey::encrypt_sk_scratch_space={}",
                scratch.available(),
                GGLWESwitchingKey::encrypt_sk_scratch_space(
                    module,
                    sk_out.n(),
                    self.basek(),
                    self.k(),
                    self.rank_in(),
                    self.rank_out()
                )
            )
        }

        let n: usize = sk_in.n().max(sk_out.n());

        let (mut sk_in_tmp, scratch1) = scratch.take_scalar_znx(n, sk_in.rank());
        (0..sk_in.rank()).for_each(|i| {
            module.vec_znx_switch_degree(
                &mut sk_in_tmp.as_vec_znx_mut(),
                i,
                &sk_in.data.as_vec_znx(),
                i,
            );
        });

        let (mut sk_out_tmp, scratch2) = scratch1.take_glwe_secret_exec(n, sk_out.rank());
        {
            let (mut tmp, _) = scratch2.take_scalar_znx(n, 1);
            (0..sk_out.rank()).for_each(|i| {
                module.vec_znx_switch_degree(&mut tmp.as_vec_znx_mut(), 0, &sk_out.data.as_vec_znx(), i);
                module.svp_prepare(&mut sk_out_tmp.data, i, &tmp, 0);
            });
        }

        self.key.encrypt_sk(
            module,
            &sk_in_tmp,
            &sk_out_tmp,
            seed_xa,
            source_xe,
            sigma,
            scratch2,
        );
        self.sk_in_n = sk_in.n();
        self.sk_out_n = sk_out.n();
    }
}
