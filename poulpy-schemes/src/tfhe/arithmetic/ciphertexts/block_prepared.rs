use poulpy_core::{
    TakeGGSW, TakeGLWECt,
    layouts::{
        GGSWInfos, GLWEInfos, LWECiphertext, LWEInfos,
        prepared::{GGSWCiphertextPrepared, GLWEToLWESwitchingKeyPrepared, Prepare},
    },
};
use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxRotate,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPrepare,
    },
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::tfhe::{
    arithmetic::{FheUintBits, UnsignedInteger},
    blind_rotation::BlindRotationAlgo,
    circuit_bootstrapping::{CircuitBootstrappingKeyPrepared, CirtuitBootstrappingExecute},
};

/// A prepared FHE ciphertext encrypting the bits of an [UnsignedInteger].
pub struct FheUintBitsPrep<D: Data, B: Backend, T: UnsignedInteger>(pub(crate) [GGSWCiphertextPrepared<D, B>; T::WORD_SIZE])
where
    [(); T::WORD_SIZE]:;

impl<D: DataMut, T: UnsignedInteger, BE: Backend> FheUintBitsPrep<D, BE, T>
where
    [(); T::WORD_SIZE]:,
{
    /// Prepares [FheUintBits] to [FheUintBitsPrep].
    pub fn prepare<BIT, LKS, CBT, BRA>(
        &mut self,
        module: &Module<BE>,
        bits: FheUintBits<BIT, T>,
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
            assert_eq!(self.0.len(), bits.0.len());
        }
        let mut lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(&bits.0[0]); //TODO: add TakeLWE
        let (mut tmp_ggsw, scratch_1) = scratch.take_ggsw(self);
        for (dst, src) in self.0.iter_mut().zip(bits.0.iter()) {
            lwe.from_glwe(module, src, lwe_ks, scratch_1);
            key.execute_to_constant(module, &mut tmp_ggsw, &lwe, 1, 1, scratch_1);
            dst.prepare(module, &tmp_ggsw, scratch_1);
        }
    }
}

impl<D: DataRef, T: UnsignedInteger, B: Backend> LWEInfos for FheUintBitsPrep<D, B, T>
where
    [(); T::WORD_SIZE]:,
{
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.0[0].base2k()
    }

    fn k(&self) -> poulpy_core::layouts::TorusPrecision {
        self.0[0].k()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.0[0].n()
    }
}

impl<D: DataRef, T: UnsignedInteger, B: Backend> GLWEInfos for FheUintBitsPrep<D, B, T>
where
    [(); T::WORD_SIZE]:,
{
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.0[0].rank()
    }
}

impl<D: DataRef, T: UnsignedInteger, B: Backend> GGSWInfos for FheUintBitsPrep<D, B, T>
where
    [(); T::WORD_SIZE]:,
{
    fn digits(&self) -> poulpy_core::layouts::Digits {
        self.0[0].digits()
    }

    fn rows(&self) -> poulpy_core::layouts::Rows {
        self.0[0].rows()
    }
}
