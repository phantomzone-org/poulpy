use core::layouts::{
    GGLWEAutomorphismKey, GGLWETensorKey, GLWECiphertext, GLWESecret, LWESecret,
    prepared::{GGLWEAutomorphismKeyExec, GGLWETensorKeyExec, GLWESecretExec},
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
    GGLWEAutomorphismKeyEncryptSkFamily, GGLWETensorKeyEncryptSkFamily, GGSWEncryptSkFamily, GLWESecretExecModuleFamily,
};

use crate::tfhe::blind_rotation::{BlindRotationKeyCGGI, BlindRotationKeyCGGIExec, BlindRotationKeyCGGIExecLayoutFamily};

pub struct CircuitBootstrappingKeyCGGI<D: Data> {
    pub(crate) brk: BlindRotationKeyCGGI<D>,
    pub(crate) tsk: GGLWETensorKey<Vec<u8>>,
    pub(crate) atk: HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>>,
}

impl CircuitBootstrappingKeyCGGI<Vec<u8>> {
    pub fn generate<DLwe, DGlwe, B: Backend>(
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
        Module<B>: GGSWEncryptSkFamily<B>
            + GLWESecretExecModuleFamily<B>
            + VecZnxAddScalarInplace
            + GGLWEAutomorphismKeyEncryptSkFamily<B>
            + VecZnxAutomorphism
            + VecZnxSwithcDegree
            + GGLWETensorKeyEncryptSkFamily<B>,
        DLwe: DataRef,
        DGlwe: DataRef,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx + TakeScalarZnx + TakeSvpPPol<B> + TakeVecZnxBig<B>,
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

        let sk_glwe_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_glwe);

        let mut brk: BlindRotationKeyCGGI<Vec<u8>> = BlindRotationKeyCGGI::alloc(
            sk_glwe.n(),
            sk_lwe.n(),
            basek,
            k_brk,
            rows_brk,
            sk_glwe.rank(),
        );
        brk.generate_from_sk(
            module,
            &sk_glwe_exec,
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

pub struct CircuitBootstrappingKeyCGGIExec<D: Data, B: Backend> {
    pub(crate) brk: BlindRotationKeyCGGIExec<D, B>,
    pub(crate) tsk: GGLWETensorKeyExec<Vec<u8>, B>,
    pub(crate) atk: HashMap<i64, GGLWEAutomorphismKeyExec<Vec<u8>, B>>,
}

impl<B: Backend> CircuitBootstrappingKeyCGGIExec<Vec<u8>, B> {
    pub fn from<DataOther: DataRef>(
        module: &Module<B>,
        other: &CircuitBootstrappingKeyCGGI<DataOther>,
        scratch: &mut Scratch<B>,
    ) -> CircuitBootstrappingKeyCGGIExec<Vec<u8>, B>
    where
        Module<B>: BlindRotationKeyCGGIExecLayoutFamily<B> + VmpPMatAlloc<B> + VmpPMatPrepare<B>,
    {
        let brk: BlindRotationKeyCGGIExec<Vec<u8>, B> = BlindRotationKeyCGGIExec::from(module, &other.brk, scratch);
        let tsk: GGLWETensorKeyExec<Vec<u8>, B> = GGLWETensorKeyExec::from(module, &other.tsk, scratch);
        let mut atk: HashMap<i64, GGLWEAutomorphismKeyExec<Vec<u8>, B>> = HashMap::new();
        for (key, value) in &other.atk {
            atk.insert(*key, GGLWEAutomorphismKeyExec::from(module, value, scratch));
        }
        CircuitBootstrappingKeyCGGIExec { brk, tsk, atk }
    }
}
