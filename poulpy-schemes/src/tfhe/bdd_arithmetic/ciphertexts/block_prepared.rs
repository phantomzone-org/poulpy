use std::marker::PhantomData;

use poulpy_core::layouts::{
    Base2K, Dsize, GGSWCiphertext, GGSWInfos, GLWEInfos, LWEInfos, Rank, Dnum, TorusPrecision, prepared::GGSWCiphertextPrepared,
};
#[cfg(test)]
use poulpy_core::{TakeGGSW, layouts::prepared::GLWESecretPrepared};
use poulpy_hal::{
    api::VmpPMatAlloc,
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};
#[cfg(test)]
use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDftInplace, TakeScalarZnx, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAlloc, VecZnxBigAllocBytes,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform,
        VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub,
        VecZnxSubInplace, VmpPrepare,
    },
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeVecZnxBigImpl, TakeVecZnxDftImpl},
    source::Source,
};

use crate::tfhe::bdd_arithmetic::{FheUintBlocks, FheUintPrepare, ToBits, UnsignedInteger};

pub(crate) struct FheUintBlocksPrepDebug<D: Data, T: UnsignedInteger> {
    pub(crate) blocks: Vec<GGSWCiphertext<D>>,
    pub(crate) _base: u8,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T: UnsignedInteger> FheUintBlocksPrepDebug<Vec<u8>, T> {
    #[allow(dead_code)]
    pub(crate) fn alloc<A, BE: Backend>(module: &Module<BE>, infos: &A) -> Self
    where
        A: GGSWInfos,
    {
        Self::alloc_with(
            module,
            infos.base2k(),
            infos.k(),
            infos.dnum(),
            infos.dsize(),
            infos.rank(),
        )
    }

    #[allow(dead_code)]
    pub(crate) fn alloc_with<BE: Backend>(
        module: &Module<BE>,
        base2k: Base2K,
        k: TorusPrecision,
        dnum: Dnum,
        dsize: Dsize,
        rank: Rank,
    ) -> Self {
        Self {
            blocks: (0..T::WORD_SIZE)
                .map(|_| GGSWCiphertext::alloc_with(module.n().into(), base2k, k, rank, dnum, dsize))
                .collect(),
            _base: 1,
            _phantom: PhantomData,
        }
    }
}

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
            infos.dnum(),
            infos.dsize(),
            infos.rank(),
        )
    }

    #[allow(dead_code)]
    pub(crate) fn alloc_with(
        module: &Module<BE>,
        base2k: Base2K,
        k: TorusPrecision,
        dnum: Dnum,
        dsize: Dsize,
        rank: Rank,
    ) -> Self
    where
        Module<BE>: VmpPMatAlloc<BE>,
    {
        Self {
            blocks: (0..T::WORD_SIZE)
                .map(|_| GGSWCiphertextPrepared::alloc_with(module, base2k, k, dnum, dsize, rank))
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
            use poulpy_core::layouts::prepared::Prepare;
            use poulpy_hal::layouts::ZnxViewMut;

            pt.at_mut(0, 0)[0] = value.bit(i) as i64;
            tmp_ggsw.encrypt_sk(&module, &pt, sk, source_xa, source_xe, scratch_2);
            self.blocks[i].prepare(module, &tmp_ggsw, scratch_2);
        }
    }

    /// Prepares [FheUintBits] to [FheUintBitsPrep].
    pub fn prepare<BIT, KEY>(&mut self, module: &Module<BE>, bits: &FheUintBlocks<BIT, T>, key: &KEY, scratch: &mut Scratch<BE>)
    where
        BIT: DataRef,
        KEY: FheUintPrepare<BE, FheUintBlocksPrep<D, BE, T>, FheUintBlocks<BIT, T>>,
    {
        key.prepare(module, self, bits, scratch);
    }
}

impl<D: DataMut, T: UnsignedInteger + ToBits> FheUintBlocksPrepDebug<D, T> {
    #[allow(dead_code)]
    #[cfg(test)]
    pub(crate) fn prepare<BIT, KEY, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        bits: &FheUintBlocks<BIT, T>,
        key: &KEY,
        scratch: &mut Scratch<BE>,
    ) where
        BIT: DataRef,
        KEY: FheUintPrepare<BE, FheUintBlocksPrepDebug<D, T>, FheUintBlocks<BIT, T>>,
    {
        key.prepare(module, self, bits, scratch);
    }
}

impl<D: DataRef, T: UnsignedInteger + ToBits> FheUintBlocksPrepDebug<D, T> {
    #[allow(dead_code)]
    #[cfg(test)]
    pub(crate) fn noise<S: DataRef, BE: Backend>(&self, module: &Module<BE>, sk: &GLWESecretPrepared<S, BE>, want: T)
    where
        Module<BE>: VecZnxDftAllocBytes
            + VecZnxBigAllocBytes
            + VecZnxDftApply<BE>
            + SvpApplyDftToDftInplace<BE>
            + VecZnxIdftApplyConsume<BE>
            + VecZnxBigAddInplace<BE>
            + VecZnxBigAddSmallInplace<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxNormalizeTmpBytes
            + VecZnxBigAlloc<BE>
            + VecZnxDftAlloc<BE>
            + VecZnxBigNormalizeTmpBytes
            + VecZnxIdftApplyTmpA<BE>
            + VecZnxAddScalarInplace
            + VecZnxSubInplace,
        BE: Backend + TakeVecZnxDftImpl<BE> + TakeVecZnxBigImpl<BE> + ScratchOwnedAllocImpl<BE> + ScratchOwnedBorrowImpl<BE>,
    {
        for (i, ggsw) in self.blocks.iter().enumerate() {
            use poulpy_hal::layouts::{ScalarZnx, ZnxViewMut};
            let mut pt_want = ScalarZnx::alloc(self.n().into(), 1);
            pt_want.at_mut(0, 0)[0] = want.bit(i) as i64;
            ggsw.print_noise(module, sk, &pt_want);
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
    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        self.blocks[0].dsize()
    }

    fn dnum(&self) -> poulpy_core::layouts::Dnum {
        self.blocks[0].dnum()
    }
}

impl<D: DataRef, T: UnsignedInteger> LWEInfos for FheUintBlocksPrepDebug<D, T> {
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

impl<D: DataRef, T: UnsignedInteger> GLWEInfos for FheUintBlocksPrepDebug<D, T> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.blocks[0].rank()
    }
}

impl<D: DataRef, T: UnsignedInteger> GGSWInfos for FheUintBlocksPrepDebug<D, T> {
    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        self.blocks[0].dsize()
    }

    fn dnum(&self) -> poulpy_core::layouts::Dnum {
        self.blocks[0].dnum()
    }
}
