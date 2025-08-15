use core::layouts::{
    GGLWEAutomorphismKey, GGLWETensorKey, GLWECiphertext, GLWESecret, LWESecret,
    prepared::{GGLWEAutomorphismKeyPrepared, GGLWETensorKeyPrepared, GLWESecretPrepared, PrepareAlloc},
};
use std::{collections::HashMap, usize};

use backend::hal::{
    api::{
        ScratchAvailable, TakeScalarZnx, TakeSvpPPol, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, VecZnxAddScalarInplace,
        VecZnxAutomorphism, VecZnxSwithcDegree, VmpPMatAlloc, VmpPMatPrepare,
    },
    layouts::{Backend, Data, DataRef, Module, Scratch},
};
use sampling::source::Source;

use core::trait_families::{
    GGLWEAutomorphismKeyEncryptSkFamily, GGLWETensorKeyEncryptSkFamily, GGSWEncryptSkFamily, GLWESecretPreparedModuleFamily,
};

use crate::tfhe::blind_rotation::{
    BlindRotationAlgo, BlindRotationKey, BlindRotationKeyAlloc, BlindRotationKeyEncryptSk, BlindRotationKeyPrepared,
};

pub trait CircuitBootstrappingKeyEncryptSk<B: Backend> {
    fn encrypt_sk<DLwe, DGlwe>(
        module: &Module<B>,
        basek: usize,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        k_brk: usize,
        rows_brk: usize,
        k_trace: usize,
        rows_trace: usize,
        k_tsk: usize,
        rows_tsk: usize,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) -> Self
    where
        DLwe: DataRef,
        DGlwe: DataRef;
}

pub struct CircuitBootstrappingKey<D: Data, BRA: BlindRotationAlgo> {
    pub(crate) brk: BlindRotationKey<D, BRA>,
    pub(crate) tsk: GGLWETensorKey<Vec<u8>>,
    pub(crate) atk: HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>>,
}

impl<BRA: BlindRotationAlgo, B: Backend> CircuitBootstrappingKeyEncryptSk<B> for CircuitBootstrappingKey<Vec<u8>, BRA>
where
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyAlloc + BlindRotationKeyEncryptSk<B>,
    Module<B>: GGSWEncryptSkFamily<B>
        + GLWESecretPreparedModuleFamily<B>
        + VecZnxAddScalarInplace
        + GGLWEAutomorphismKeyEncryptSkFamily<B>
        + VecZnxAutomorphism
        + VecZnxSwithcDegree
        + GGLWETensorKeyEncryptSkFamily<B>,
    Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx + TakeScalarZnx + TakeSvpPPol<B> + TakeVecZnxBig<B>,
{
    fn encrypt_sk<DLwe, DGlwe>(
        module: &Module<B>,
        basek: usize,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        k_brk: usize,
        rows_brk: usize,
        k_trace: usize,
        rows_trace: usize,
        k_tsk: usize,
        rows_tsk: usize,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) -> Self
    where
        DLwe: DataRef,
        DGlwe: DataRef,
    {
        let mut auto_keys: HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>> = HashMap::new();
        let gal_els: Vec<i64> = GLWECiphertext::trace_galois_elements(&module);
        gal_els.iter().for_each(|gal_el| {
            let mut key: GGLWEAutomorphismKey<Vec<u8>> =
                GGLWEAutomorphismKey::alloc(sk_glwe.n(), basek, k_trace, rows_trace, 1, sk_glwe.rank());
            key.encrypt_sk(
                &module, *gal_el, &sk_glwe, source_xa, source_xe, sigma, scratch,
            );
            auto_keys.insert(*gal_el, key);
        });

        let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_glwe.prepare_alloc(module, scratch);

        let mut brk: BlindRotationKey<Vec<u8>, BRA> = BlindRotationKey::<Vec<u8>, BRA>::alloc(
            sk_glwe.n(),
            sk_lwe.n(),
            basek,
            k_brk,
            rows_brk,
            sk_glwe.rank(),
        );

        brk.encrypt_sk(
            module,
            &sk_glwe_prepared,
            sk_lwe,
            source_xa,
            source_xe,
            sigma,
            scratch,
        );

        let mut tsk: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(sk_glwe.n(), basek, k_tsk, rows_tsk, 1, sk_glwe.rank());
        tsk.encrypt_sk(module, &sk_glwe, source_xa, source_xe, sigma, scratch);

        Self {
            brk,
            atk: auto_keys,
            tsk,
        }
    }
}

pub struct CircuitBootstrappingKeyPrepared<D: Data, BRA: BlindRotationAlgo, B: Backend> {
    pub(crate) brk: BlindRotationKeyPrepared<D, BRA, B>,
    pub(crate) tsk: GGLWETensorKeyPrepared<Vec<u8>, B>,
    pub(crate) atk: HashMap<i64, GGLWEAutomorphismKeyPrepared<Vec<u8>, B>>,
}

impl<D: DataRef, BRA: BlindRotationAlgo, B: Backend> PrepareAlloc<B, CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, B>>
    for CircuitBootstrappingKey<D, BRA>
where
    Module<B>: VmpPMatAlloc<B> + VmpPMatPrepare<B>,
    BlindRotationKey<D, BRA>: PrepareAlloc<B, BlindRotationKeyPrepared<Vec<u8>, BRA, B>>,
    GGLWETensorKey<D>: PrepareAlloc<B, GGLWETensorKeyPrepared<Vec<u8>, B>>,
    GGLWEAutomorphismKey<D>: PrepareAlloc<B, GGLWEAutomorphismKeyPrepared<Vec<u8>, B>>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, B> {
        let brk: BlindRotationKeyPrepared<Vec<u8>, BRA, B> = self.brk.prepare_alloc(module, scratch);
        let tsk: GGLWETensorKeyPrepared<Vec<u8>, B> = self.tsk.prepare_alloc(module, scratch);
        let mut atk: HashMap<i64, GGLWEAutomorphismKeyPrepared<Vec<u8>, B>> = HashMap::new();
        for (key, value) in &self.atk {
            atk.insert(*key, value.prepare_alloc(module, scratch));
        }
        CircuitBootstrappingKeyPrepared { brk, tsk, atk }
    }
}
