use poulpy_core::layouts::{
    AutomorphismKey, AutomorphismKeyLayout, GGLWEInfos, GGSWInfos, GLWE, GLWEInfos, GLWESecret, LWEInfos, LWESecret, TensorKey,
    TensorKeyLayout,
    prepared::{AutomorphismKeyPrepared, GLWESecretPrepared, PrepareAlloc, TensorKeyPrepared},
};
use std::collections::HashMap;

use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare, TakeScalarZnx,
        TakeSvpPPol, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace,
        VecZnxAutomorphism, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxFillUniform, VecZnxIdftApplyConsume,
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
    fn brk_infos(&self) -> BlindRotationKeyLayout;
    fn atk_infos(&self) -> AutomorphismKeyLayout;
    fn tsk_infos(&self) -> TensorKeyLayout;
}

#[derive(Debug, Clone, Copy)]
pub struct CircuitBootstrappingKeyLayout {
    pub layout_brk: BlindRotationKeyLayout,
    pub layout_atk: AutomorphismKeyLayout,
    pub layout_tsk: TensorKeyLayout,
}

impl CircuitBootstrappingKeyInfos for CircuitBootstrappingKeyLayout {
    fn atk_infos(&self) -> AutomorphismKeyLayout {
        self.layout_atk
    }

    fn brk_infos(&self) -> BlindRotationKeyLayout {
        self.layout_brk
    }

    fn tsk_infos(&self) -> TensorKeyLayout {
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
    pub(crate) tsk: TensorKey<Vec<u8>>,
    pub(crate) atk: HashMap<i64, AutomorphismKey<Vec<u8>>>,
}

impl<BRA: BlindRotationAlgo, B: Backend> CircuitBootstrappingKeyEncryptSk<B> for CircuitBootstrappingKey<Vec<u8>, BRA>
where
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyAlloc + BlindRotationKeyEncryptSk<B>,
    Module<B>: SvpApplyDftToDft<B>
        + VecZnxIdftApplyTmpA<B>
        + VecZnxAddScalarInplace
        + VecZnxDftBytesOf
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
        + SvpPPolBytesOf
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
    {
        assert_eq!(sk_lwe.n(), cbt_infos.brk_infos().n_lwe());
        assert_eq!(sk_glwe.n(), cbt_infos.brk_infos().n_glwe());
        assert_eq!(sk_glwe.n(), cbt_infos.atk_infos().n());
        assert_eq!(sk_glwe.n(), cbt_infos.tsk_infos().n());

        let atk_infos: AutomorphismKeyLayout = cbt_infos.atk_infos();
        let brk_infos: BlindRotationKeyLayout = cbt_infos.brk_infos();
        let trk_infos: TensorKeyLayout = cbt_infos.tsk_infos();

        let mut auto_keys: HashMap<i64, AutomorphismKey<Vec<u8>>> = HashMap::new();
        let gal_els: Vec<i64> = GLWE::trace_galois_elements(module);
        gal_els.iter().for_each(|gal_el| {
            let mut key: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_from_infos(&atk_infos);
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

        let mut tsk: TensorKey<Vec<u8>> = TensorKey::alloc_from_infos(&trk_infos);
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
    pub(crate) tsk: TensorKeyPrepared<Vec<u8>, B>,
    pub(crate) atk: HashMap<i64, AutomorphismKeyPrepared<Vec<u8>, B>>,
}

impl<D: DataRef, BRA: BlindRotationAlgo, B: Backend> CircuitBootstrappingKeyInfos for CircuitBootstrappingKeyPrepared<D, BRA, B> {
    fn atk_infos(&self) -> AutomorphismKeyLayout {
        let (_, atk) = self.atk.iter().next().expect("atk is empty");
        AutomorphismKeyLayout {
            n: atk.n(),
            base2k: atk.base2k(),
            k: atk.k(),
            dnum: atk.dnum(),
            dsize: atk.dsize(),
            rank: atk.rank(),
        }
    }

    fn brk_infos(&self) -> BlindRotationKeyLayout {
        BlindRotationKeyLayout {
            n_glwe: self.brk.n_glwe(),
            n_lwe: self.brk.n_lwe(),
            base2k: self.brk.base2k(),
            k: self.brk.k(),
            dnum: self.brk.dnum(),
            rank: self.brk.rank(),
        }
    }

    fn tsk_infos(&self) -> TensorKeyLayout {
        TensorKeyLayout {
            n: self.tsk.n(),
            base2k: self.tsk.base2k(),
            k: self.tsk.k(),
            dnum: self.tsk.dnum(),
            dsize: self.tsk.dsize(),
            rank: self.tsk.rank(),
        }
    }
}

impl<D: DataRef, BRA: BlindRotationAlgo, B: Backend> PrepareAlloc<B, CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, B>>
    for CircuitBootstrappingKey<D, BRA>
where
    Module<B>: VmpPMatAlloc<B> + VmpPrepare<B>,
    BlindRotationKey<D, BRA>: PrepareAlloc<B, BlindRotationKeyPrepared<Vec<u8>, BRA, B>>,
    TensorKey<D>: PrepareAlloc<B, TensorKeyPrepared<Vec<u8>, B>>,
    AutomorphismKey<D>: PrepareAlloc<B, AutomorphismKeyPrepared<Vec<u8>, B>>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, B> {
        let brk: BlindRotationKeyPrepared<Vec<u8>, BRA, B> = self.brk.prepare_alloc(module, scratch);
        let tsk: TensorKeyPrepared<Vec<u8>, B> = self.tsk.prepare_alloc(module, scratch);
        let mut atk: HashMap<i64, AutomorphismKeyPrepared<Vec<u8>, B>> = HashMap::new();
        for (key, value) in &self.atk {
            atk.insert(*key, value.prepare_alloc(module, scratch));
        }
        CircuitBootstrappingKeyPrepared { brk, tsk, atk }
    }
}
