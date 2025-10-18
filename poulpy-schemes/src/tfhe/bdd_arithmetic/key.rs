#[cfg(test)]
use crate::tfhe::bdd_arithmetic::FheUintBlocksPrepDebug;
use crate::tfhe::{
    bdd_arithmetic::{FheUintBlocks, FheUintBlocksPrep, UnsignedInteger},
    blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyAlloc, BlindRotationKeyEncryptSk},
    circuit_bootstrapping::{
        CircuitBootstrappingKey, CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyLayout,
        CircuitBootstrappingKeyPrepared, CirtuitBootstrappingExecute,
    },
};
use poulpy_core::{
    TakeGGSW, TakeGLWE,
    layouts::{
        GLWESecret, GLWEToLWEKeyLayout, GLWEToLWESwitchingKey, LWE, LWESecret,
        prepared::{GLWEToLWESwitchingKeyPrepared, Prepare, PrepareAlloc},
    },
};
use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare, TakeScalarZnx,
        TakeSvpPPol, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace,
        VecZnxAutomorphism, VecZnxAutomorphismInplace, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxSub, VecZnxSubInplace, VecZnxSwitchRing,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPrepare,
    },
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
    source::Source,
};

pub trait BDDKeyInfos {
    fn cbt_infos(&self) -> CircuitBootstrappingKeyLayout;
    fn ks_infos(&self) -> GLWEToLWEKeyLayout;
}

#[derive(Debug, Clone, Copy)]
pub struct BDDKeyLayout {
    pub cbt: CircuitBootstrappingKeyLayout,
    pub ks: GLWEToLWEKeyLayout,
}

impl BDDKeyInfos for BDDKeyLayout {
    fn cbt_infos(&self) -> CircuitBootstrappingKeyLayout {
        self.cbt
    }

    fn ks_infos(&self) -> GLWEToLWEKeyLayout {
        self.ks
    }
}

pub struct BDDKey<CBT, LWE, BRA>
where
    CBT: Data,
    LWE: Data,
    BRA: BlindRotationAlgo,
{
    cbt: CircuitBootstrappingKey<CBT, BRA>,
    ks: GLWEToLWESwitchingKey<LWE>,
}

impl<BRA: BlindRotationAlgo> BDDKey<Vec<u8>, Vec<u8>, BRA> {
    pub fn encrypt_sk<DLwe, DGlwe, A, BE: Backend>(
        module: &Module<BE>,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        infos: &A,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) -> Self
    where
        A: BDDKeyInfos,
        DLwe: DataRef,
        DGlwe: DataRef,
        BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyAlloc + BlindRotationKeyEncryptSk<BE>,
        Module<BE>: SvpApplyDftToDft<BE>
            + VecZnxIdftApplyTmpA<BE>
            + VecZnxAddScalarInplace
            + VecZnxDftBytesOf
            + VecZnxBigNormalize<BE>
            + VecZnxDftApply<BE>
            + SvpApplyDftToDftInplace<BE>
            + VecZnxIdftApplyConsume<BE>
            + VecZnxNormalizeTmpBytes
            + VecZnxFillUniform
            + VecZnxSubInplace
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<BE>
            + VecZnxAddNormal
            + VecZnxNormalize<BE>
            + VecZnxSub
            + SvpPrepare<BE>
            + VecZnxSwitchRing
            + SvpPPolBytesOf
            + SvpPPolAlloc<BE>
            + VecZnxAutomorphism
            + VecZnxAutomorphismInplace<BE>,
        Scratch<BE>: TakeVecZnxDft<BE> + ScratchAvailable + TakeVecZnx + TakeScalarZnx + TakeSvpPPol<BE> + TakeVecZnxBig<BE>,
    {
        let mut ks: GLWEToLWESwitchingKey<Vec<u8>> = GLWEToLWESwitchingKey::alloc(&infos.ks_infos());
        ks.encrypt_sk(module, sk_lwe, sk_glwe, source_xa, source_xe, scratch);

        Self {
            cbt: CircuitBootstrappingKey::encrypt_sk(
                module,
                sk_lwe,
                sk_glwe,
                &infos.cbt_infos(),
                source_xa,
                source_xe,
                scratch,
            ),
            ks,
        }
    }
}

pub struct BDDKeyPrepared<CBT, LWE, BRA, BE>
where
    CBT: Data,
    LWE: Data,
    BRA: BlindRotationAlgo,
    BE: Backend,
{
    cbt: CircuitBootstrappingKeyPrepared<CBT, BRA, BE>,
    ks: GLWEToLWESwitchingKeyPrepared<LWE, BE>,
}

impl<CBT: DataMut, LWE: DataMut, BRA: BlindRotationAlgo, BE: Backend> PrepareAlloc<BE, BDDKeyPrepared<CBT, LWE, BRA, BE>>
    for BDDKey<CBT, LWE, BRA>
where
    CircuitBootstrappingKey<CBT, BRA>: PrepareAlloc<BE, CircuitBootstrappingKeyPrepared<CBT, BRA, BE>>,
    GLWEToLWESwitchingKey<LWE>: PrepareAlloc<BE, GLWEToLWESwitchingKeyPrepared<LWE, BE>>,
{
    fn prepare_alloc(&self, module: &Module<BE>, scratch: &mut Scratch<BE>) -> BDDKeyPrepared<CBT, LWE, BRA, BE> {
        BDDKeyPrepared {
            cbt: self.cbt.prepare_alloc(module, scratch),
            ks: self.ks.prepare_alloc(module, scratch),
        }
    }
}

pub trait FheUintPrepare<BE: Backend, OUT, IN> {
    fn prepare(&self, module: &Module<BE>, out: &mut OUT, bits: &IN, scratch: &mut Scratch<BE>);
}

impl<CBT, OUT, IN, LWE, BRA, BE, T> FheUintPrepare<BE, FheUintBlocksPrep<OUT, BE, T>, FheUintBlocks<IN, T>>
    for BDDKeyPrepared<CBT, LWE, BRA, BE>
where
    T: UnsignedInteger,
    CBT: DataRef,
    OUT: DataMut,
    IN: DataRef,
    LWE: DataRef,
    BRA: BlindRotationAlgo,
    BE: Backend,
    Module<BE>: VmpPrepare<BE>
        + VecZnxRotate
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddSmallInplace<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    Scratch<BE>: ScratchAvailable + TakeVecZnxDft<BE> + TakeGLWE + TakeVecZnx + TakeGGSW,
    CircuitBootstrappingKeyPrepared<CBT, BRA, BE>: CirtuitBootstrappingExecute<BE>,
{
    fn prepare(
        &self,
        module: &Module<BE>,
        out: &mut FheUintBlocksPrep<OUT, BE, T>,
        bits: &FheUintBlocks<IN, T>,
        scratch: &mut Scratch<BE>,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(out.blocks.len(), bits.blocks.len());
        }
        let mut lwe: LWE<Vec<u8>> = LWE::alloc(&bits.blocks[0]); //TODO: add TakeLWE
        let (mut tmp_ggsw, scratch_1) = scratch.take_ggsw(out);
        for (dst, src) in out.blocks.iter_mut().zip(bits.blocks.iter()) {
            lwe.from_glwe(module, src, &self.ks, scratch_1);
            self.cbt
                .execute_to_constant(module, &mut tmp_ggsw, &lwe, 1, 1, scratch_1);
            dst.prepare(module, &tmp_ggsw, scratch_1);
        }
    }
}

#[cfg(test)]
impl<CBT, OUT, IN, LWE, BRA, BE, T> FheUintPrepare<BE, FheUintBlocksPrepDebug<OUT, T>, FheUintBlocks<IN, T>>
    for BDDKeyPrepared<CBT, LWE, BRA, BE>
where
    T: UnsignedInteger,
    CBT: DataRef,
    OUT: DataMut,
    IN: DataRef,
    LWE: DataRef,
    BRA: BlindRotationAlgo,
    BE: Backend,
    Module<BE>: VmpPrepare<BE>
        + VecZnxRotate
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAdd<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddSmallInplace<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    Scratch<BE>: ScratchAvailable + TakeVecZnxDft<BE> + TakeGLWE + TakeVecZnx + TakeGGSW,
    CircuitBootstrappingKeyPrepared<CBT, BRA, BE>: CirtuitBootstrappingExecute<BE>,
{
    fn prepare(
        &self,
        module: &Module<BE>,
        out: &mut FheUintBlocksPrepDebug<OUT, T>,
        bits: &FheUintBlocks<IN, T>,
        scratch: &mut Scratch<BE>,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(out.blocks.len(), bits.blocks.len());
        }
        let mut lwe: LWE<Vec<u8>> = LWE::alloc(&bits.blocks[0]); //TODO: add TakeLWE
        for (dst, src) in out.blocks.iter_mut().zip(bits.blocks.iter()) {
            lwe.from_glwe(module, src, &self.ks, scratch);
            self.cbt
                .execute_to_constant(module, dst, &lwe, 1, 1, scratch);
        }
    }
}
