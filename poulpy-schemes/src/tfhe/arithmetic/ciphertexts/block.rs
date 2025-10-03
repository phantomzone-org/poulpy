use poulpy_core::layouts::{GLWECiphertext, GLWEInfos, LWEInfos};
use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxBigNormalize, VecZnxCopy, VecZnxDftAllocBytes,
        VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxSub,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::tfhe::arithmetic::{ADD_OP32, FheUintBitsPrep, UnsignedInteger};

/// An FHE ciphertext encrypting the bits of an [UnsignedInteger].
pub struct FheUintBits<D: Data, T: UnsignedInteger>(pub(crate) [GLWECiphertext<D>; T::WORD_SIZE])
where
    [(); T::WORD_SIZE]:;

impl<D: DataRef, T: UnsignedInteger> LWEInfos for FheUintBits<D, T>
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

impl<D: DataRef, T: UnsignedInteger> GLWEInfos for FheUintBits<D, T>
where
    [(); T::WORD_SIZE]:,
{
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.0[0].rank()
    }
}

impl<D: DataMut, T: UnsignedInteger> FheUintBits<D, T>
where
    [(); T::WORD_SIZE]:,
{
    pub fn add<A, B, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &FheUintBitsPrep<A, BE, T>,
        b: &FheUintBitsPrep<B, BE, T>,
        scratch: &mut Scratch<BE>,
    ) where
        A: DataRef,
        B: DataRef,
        Module<BE>: VecZnxSub
            + VecZnxCopy
            + VecZnxNegateInplace
            + VecZnxDftAllocBytes
            + VecZnxAddInplace
            + VmpApplyDftToDftTmpBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftApply<BE>
            + VmpApplyDftToDft<BE>
            + VmpApplyDftToDftAdd<BE>
            + VecZnxIdftApplyConsume<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxNormalize<BE>,
        Scratch<BE>: TakeVecZnxDft<BE> + ScratchAvailable + TakeVecZnx,
    {
        match T::WORD_SIZE {
            32 => ADD_OP32.execute(module, &mut self.0, &a.0, &b.0, scratch),
            _ => unimplemented!(),
        }
    }
}
