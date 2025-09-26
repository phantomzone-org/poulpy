use poulpy_core::layouts::{
    GGLWEAutomorphismKey, GGLWEAutomorphismKeyLayout, GGLWELayoutInfos, GGLWETensorKey, GGLWETensorKeyLayout, GGSWInfos,
    GLWECiphertext, GLWEInfos, GLWESecret, LWEInfos, LWESecret,
    prepared::{GGLWEAutomorphismKeyPrepared, GGLWETensorKeyPrepared, GLWESecretPrepared, PrepareAlloc},
};
use std::collections::HashMap;

use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, TakeScalarZnx,
        TakeSvpPPol, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace,
        VecZnxAutomorphism, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume,
        VecZnxIdftApplyTmpA, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace,
        VecZnxSwitchRing, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Data, DataRef, Module, Scratch},
    source::Source,
};

use crate::tfhe::blind_rotation::{
    BlindRotationAlgo, BlindRotationKey, BlindRotationKeyAlloc, BlindRotationKeyEncryptSk, BlindRotationKeyInfos,
    BlindRotationKeyLayout, BlindRotationKeyPrepared,
};

pub trait CircuitBootstrappingKeyInfos {
    fn layout_brk(&self) -> BlindRotationKeyLayout;
    fn layout_atk(&self) -> GGLWEAutomorphismKeyLayout;
    fn layout_tsk(&self) -> GGLWETensorKeyLayout;
}

pub struct CircuitBootstrappingKeyLayout {
    pub layout_brk: BlindRotationKeyLayout,
    pub layout_atk: GGLWEAutomorphismKeyLayout,
    pub layout_tsk: GGLWETensorKeyLayout,
}

impl CircuitBootstrappingKeyInfos for CircuitBootstrappingKeyLayout {
    fn layout_atk(&self) -> GGLWEAutomorphismKeyLayout {
        self.layout_atk
    }

    fn layout_brk(&self) -> BlindRotationKeyLayout {
        self.layout_brk
    }

    fn layout_tsk(&self) -> GGLWETensorKeyLayout {
        self.layout_tsk
    }
}

pub trait CircuitBootstrappingKeyEncryptSk<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn encrypt_sk<DLwe, DGlwe, INFOS>(
        module: &Module<B>,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        cbt_infos: &INFOS,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) -> Self
    where
        INFOS: CircuitBootstrappingKeyInfos,
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
    Module<B>: SvpApplyDftToDft<B>
        + VecZnxIdftApplyTmpA<B>
        + VecZnxAddScalarInplace
        + VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxSubInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub
        + SvpPrepare<B>
        + VecZnxSwitchRing
        + SvpPPolAllocBytes
        + SvpPPolAlloc<B>
        + VecZnxAutomorphism,
    Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx + TakeScalarZnx + TakeSvpPPol<B> + TakeVecZnxBig<B>,
{
    fn encrypt_sk<DLwe, DGlwe, INFOS>(
        module: &Module<B>,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        cbt_infos: &INFOS,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) -> Self
    where
        INFOS: CircuitBootstrappingKeyInfos,
        DLwe: DataRef,
        DGlwe: DataRef,
        Module<B>:,
    {
        assert_eq!(sk_lwe.n(), cbt_infos.layout_brk().n_lwe());
        assert_eq!(sk_glwe.n(), cbt_infos.layout_brk().n_glwe());
        assert_eq!(sk_glwe.n(), cbt_infos.layout_atk().n());
        assert_eq!(sk_glwe.n(), cbt_infos.layout_tsk().n());

        let atk_infos: GGLWEAutomorphismKeyLayout = cbt_infos.layout_atk();
        let brk_infos: BlindRotationKeyLayout = cbt_infos.layout_brk();
        let trk_infos: GGLWETensorKeyLayout = cbt_infos.layout_tsk();

        let mut auto_keys: HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>> = HashMap::new();
        let gal_els: Vec<i64> = GLWECiphertext::trace_galois_elements(module);
        gal_els.iter().for_each(|gal_el| {
            let mut key: GGLWEAutomorphismKey<Vec<u8>> = GGLWEAutomorphismKey::alloc(&atk_infos);
            key.encrypt_sk(module, *gal_el, sk_glwe, source_xa, source_xe, scratch);
            auto_keys.insert(*gal_el, key);
        });

        let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_glwe.prepare_alloc(module, scratch);

        let mut brk: BlindRotationKey<Vec<u8>, BRA> = BlindRotationKey::<Vec<u8>, BRA>::alloc(&brk_infos);
        brk.encrypt_sk(
            module,
            &sk_glwe_prepared,
            sk_lwe,
            source_xa,
            source_xe,
            scratch,
        );

        let mut tsk: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(&trk_infos);
        tsk.encrypt_sk(module, sk_glwe, source_xa, source_xe, scratch);

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

impl<D: DataRef, BRA: BlindRotationAlgo, B: Backend> CircuitBootstrappingKeyInfos for CircuitBootstrappingKeyPrepared<D, BRA, B> {
    fn layout_atk(&self) -> GGLWEAutomorphismKeyLayout {
        let (_, atk) = self.atk.iter().next().expect("atk is empty");
        GGLWEAutomorphismKeyLayout {
            n: atk.n(),
            base2k: atk.base2k(),
            k: atk.k(),
            rows: atk.rows(),
            digits: atk.digits(),
            rank: atk.rank(),
        }
    }

    fn layout_brk(&self) -> BlindRotationKeyLayout {
        BlindRotationKeyLayout {
            n_glwe: self.brk.n_glwe(),
            n_lwe: self.brk.n_lwe(),
            base2k: self.brk.base2k(),
            k: self.brk.k(),
            rows: self.brk.rows(),
            rank: self.brk.rank(),
        }
    }

    fn layout_tsk(&self) -> GGLWETensorKeyLayout {
        GGLWETensorKeyLayout {
            n: self.tsk.n(),
            base2k: self.tsk.base2k(),
            k: self.tsk.k(),
            rows: self.tsk.rows(),
            digits: self.tsk.digits(),
            rank: self.tsk.rank(),
        }
    }
}

impl<D: DataRef, BRA: BlindRotationAlgo, B: Backend> PrepareAlloc<B, CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, B>>
    for CircuitBootstrappingKey<D, BRA>
where
    Module<B>: VmpPMatAlloc<B> + VmpPrepare<B>,
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
