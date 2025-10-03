use std::marker::PhantomData;

use itertools::Itertools;
use poulpy_core::layouts::{Base2K, GLWECiphertext, GLWEInfos, GLWEPlaintextLayout, LWEInfos, Rank, TorusPrecision};

use poulpy_core::{TakeGLWEPt, layouts::prepared::GLWESecretPrepared};
use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, VecZnxAddInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxCopy, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxSub, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes
    },
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use poulpy_hal::{
    api::{SvpApplyDftToDftInplace, VecZnxAddNormal, VecZnxFillUniform, VecZnxNormalizeInplace, VecZnxSubInplace},
    source::Source,
};

use crate::tfhe::bdd_arithmetic::{FheUintBlocksPrep, FromBits, ToBits, UnsignedInteger, ADD_OP32};

/// An FHE ciphertext encrypting the bits of an [UnsignedInteger].
pub struct FheUintBlocks<D: Data, T: UnsignedInteger> {
    pub(crate) blocks: Vec<GLWECiphertext<D>>,
    pub(crate) _base: u8,
    pub(crate) _phantom: PhantomData<T>,
}

impl<D: DataRef, T: UnsignedInteger> LWEInfos for FheUintBlocks<D, T> {
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

impl<D: DataRef, T: UnsignedInteger> GLWEInfos for FheUintBlocks<D, T> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.blocks[0].rank()
    }
}

impl<T: UnsignedInteger> FheUintBlocks<Vec<u8>, T> {
    #[allow(dead_code)]
    pub(crate) fn alloc<A, BE: Backend>(module: &Module<BE>, infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc_with(module, infos.base2k(), infos.k(), infos.rank())
    }

    #[allow(dead_code)]
    pub(crate) fn alloc_with<BE: Backend>(module: &Module<BE>, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        Self {
            blocks: (0..T::WORD_SIZE)
                .map(|_| GLWECiphertext::alloc_with(module.n().into(), base2k, k, rank))
                .collect(),
            _base: 1,
            _phantom: PhantomData,
        }
    }
}

impl<D: DataMut, T: UnsignedInteger + ToBits> FheUintBlocks<D, T> {
    #[allow(dead_code)]
    #[cfg(test)]
    pub(crate) fn encrypt_sk<S, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        value: T,
        sk: &GLWESecretPrepared<S, BE>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S: DataRef,
        Module<BE>: VecZnxDftAllocBytes
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
            + VecZnxSub,
        Scratch<BE>: TakeVecZnxDft<BE> + ScratchAvailable + TakeVecZnx + TakeGLWEPt<BE>,
    {
        use poulpy_core::layouts::GLWEPlaintextLayout;

        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::WORD_SIZE));
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(sk.n(), module.n() as u32);
        }

        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 1_usize.into(),
        };

        let (mut pt, scratch_1) = scratch.take_glwe_pt(&pt_infos);

        for i in 0..T::WORD_SIZE {
            pt.encode_coeff_i64(value.bit(i) as i64, TorusPrecision(1), 0);
            self.blocks[i].encrypt_sk(&module, &pt, sk, source_xa, source_xe, scratch_1);
        }
    }

    pub fn add<A, B, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &FheUintBlocksPrep<A, BE, T>,
        b: &FheUintBlocksPrep<B, BE, T>,
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
            32 => ADD_OP32.execute(module, &mut self.blocks, &a.blocks, &b.blocks, scratch),
            _ => unimplemented!(),
        }
    }
}

impl<D: DataRef, T: UnsignedInteger + FromBits> FheUintBlocks<D, T> {
    pub fn decrypt<S: DataRef, BE: Backend>(
        &self,
        module: &Module<BE>,
        sk: &GLWESecretPrepared<S, BE>,
        scratch: &mut Scratch<BE>,
    ) -> T
    where
        Module<BE>: VecZnxDftApply<BE>
            + SvpApplyDftToDftInplace<BE>
            + VecZnxIdftApplyConsume<BE>
            + VecZnxBigAddInplace<BE>
            + VecZnxBigAddSmallInplace<BE>
            + VecZnxBigNormalize<BE>,
        Scratch<BE>: TakeVecZnxDft<BE> + TakeVecZnxBig<BE> + TakeGLWEPt<BE>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::WORD_SIZE));
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(sk.n(), module.n() as u32);
        }

        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 1_usize.into(),
        };

        let (mut pt, scratch_1) = scratch.take_glwe_pt(&pt_infos);

        let mut bits: Vec<u8> = vec![0u8; T::WORD_SIZE];

        for i in 0..T::WORD_SIZE{
            self.blocks[i].decrypt(module, &mut pt, sk, scratch_1);
            bits[i] = pt.decode_coeff_i64(TorusPrecision(1), 0) as u8;
        }

        T::from_bits(&bits)
    }
}