use std::marker::PhantomData;

#[cfg(test)]
use poulpy_core::layouts::prepared::GLWESecretPrepared;
use poulpy_core::{
    TakeGGSW, TakeGLWECt,
    layouts::{
        Base2K, Digits, GGSWInfos, GLWEInfos, LWECiphertext, LWEInfos, Rank, Rows, TorusPrecision,
        prepared::{GGSWCiphertextPrepared, GLWEToLWESwitchingKeyPrepared, Prepare},
    },
};
use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxRotate,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};
#[cfg(test)]
use poulpy_hal::{
    api::{
        SvpApplyDftToDftInplace, TakeScalarZnx, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxFillUniform,
        VecZnxNormalizeInplace, VecZnxSub, VecZnxSubInplace,
    },
    source::Source,
};

use crate::tfhe::{
    bdd_arithmetic::{FheUintBlocks, ToBits, UnsignedInteger},
    blind_rotation::BlindRotationAlgo,
    circuit_bootstrapping::{CircuitBootstrappingKeyPrepared, CirtuitBootstrappingExecute},
};

/// A prepared FHE ciphertext encrypting the bits of an [UnsignedInteger].
pub struct FheUintBlocksPrep<D: Data, B: Backend, T: UnsignedInteger> {
    pub(crate) blocks: Vec<GGSWCiphertextPrepared<D, B>>,
    pub(crate) _base: u8,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T: UnsignedInteger, BE: Backend> FheUintBlocksPrep<Vec<u8>, BE, T>
where
    Module<BE>: VmpPMatAlloc<BE>,
{
    #[allow(dead_code)]
    pub(crate) fn alloc<A>(module: &Module<BE>, infos: &A) -> Self
    where
        A: GGSWInfos,
    {
        Self::alloc_with(
            module,
            infos.base2k(),
            infos.k(),
            infos.rows(),
            infos.digits(),
            infos.rank(),
        )
    }

    #[allow(dead_code)]
    pub(crate) fn alloc_with(
        module: &Module<BE>,
        base2k: Base2K,
        k: TorusPrecision,
        rows: Rows,
        digits: Digits,
        rank: Rank,
    ) -> Self
    where
        Module<BE>: VmpPMatAlloc<BE>,
    {
        Self {
            blocks: (0..T::WORD_SIZE)
                .map(|_| GGSWCiphertextPrepared::alloc_with(module, base2k, k, rows, digits, rank))
                .collect(),
            _base: 1,
            _phantom: PhantomData,
        }
    }
}

impl<D: DataMut, T: UnsignedInteger + ToBits, BE: Backend> FheUintBlocksPrep<D, BE, T> {
    #[allow(dead_code)]
    #[cfg(test)]
    pub(crate) fn encrypt_sk<S>(
        &mut self,
        module: &Module<BE>,
        value: T,
        sk: &GLWESecretPrepared<S, BE>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S: DataRef,
        Module<BE>: VecZnxAddScalarInplace
            + VecZnxDftAllocBytes
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
            + VmpPrepare<BE>,
        Scratch<BE>: TakeVecZnxDft<BE> + ScratchAvailable + TakeVecZnx + TakeGGSW + TakeScalarZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::WORD_SIZE));
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(sk.n(), module.n() as u32);
        }

        let (mut tmp_ggsw, scratch_1) = scratch.take_ggsw(self);
        let (mut pt, scratch_2) = scratch_1.take_scalar_znx(module.n(), 1);

        for i in 0..T::WORD_SIZE {
            use poulpy_hal::layouts::ZnxViewMut;

            pt.at_mut(0, 0)[0] = value.bit(i) as i64;
            tmp_ggsw.encrypt_sk(&module, &pt, sk, source_xa, source_xe, scratch_2);
            self.blocks[i].prepare(module, &tmp_ggsw, scratch_2);
        }
    }

    /// Prepares [FheUintBits] to [FheUintBitsPrep].
    pub fn prepare<BIT, LKS, CBT, BRA>(
        &mut self,
        module: &Module<BE>,
        bits: FheUintBlocks<BIT, T>,
        lwe_ks: &GLWEToLWESwitchingKeyPrepared<LKS, BE>,
        key: &CircuitBootstrappingKeyPrepared<CBT, BRA, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        BIT: DataRef,
        LKS: DataRef,
        CBT: DataRef,
        BRA: BlindRotationAlgo,
        Module<BE>: VmpPrepare<BE>
            + VecZnxRotate
            + VecZnxDftAllocBytes
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
        Scratch<BE>: ScratchAvailable + TakeVecZnxDft<BE> + TakeGLWECt + TakeVecZnx + TakeGGSW,
        CircuitBootstrappingKeyPrepared<CBT, BRA, BE>: CirtuitBootstrappingExecute<BE>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.blocks.len(), bits.blocks.len());
        }
        let mut lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(&bits.blocks[0]); //TODO: add TakeLWE
        let (mut tmp_ggsw, scratch_1) = scratch.take_ggsw(self);
        for (dst, src) in self.blocks.iter_mut().zip(bits.blocks.iter()) {
            lwe.from_glwe(module, src, lwe_ks, scratch_1);
            key.execute_to_constant(module, &mut tmp_ggsw, &lwe, 1, 1, scratch_1);
            dst.prepare(module, &tmp_ggsw, scratch_1);
        }
    }
}

impl<D: DataRef, T: UnsignedInteger, B: Backend> LWEInfos for FheUintBlocksPrep<D, B, T> {
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.blocks[0].base2k()
    }

    fn k(&self) -> poulpy_core::layouts::TorusPrecision {
        self.blocks[0].k()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.blocks[0].n()
    }
}

impl<D: DataRef, T: UnsignedInteger, B: Backend> GLWEInfos for FheUintBlocksPrep<D, B, T> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.blocks[0].rank()
    }
}

impl<D: DataRef, T: UnsignedInteger, B: Backend> GGSWInfos for FheUintBlocksPrep<D, B, T> {
    fn digits(&self) -> poulpy_core::layouts::Digits {
        self.blocks[0].digits()
    }

    fn rows(&self) -> poulpy_core::layouts::Rows {
        self.blocks[0].rows()
    }
}
