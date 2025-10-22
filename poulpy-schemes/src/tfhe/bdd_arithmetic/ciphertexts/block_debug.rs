use std::marker::PhantomData;

use crate::tfhe::bdd_arithmetic::{BDDKeyPrepared, FheUintBlockDebugPrepare, ToBits};
use crate::tfhe::{
    bdd_arithmetic::{FheUintBlocks, UnsignedInteger},
    blind_rotation::BlindRotationAlgo,
    circuit_bootstrapping::CirtuitBootstrappingExecute,
};
use poulpy_core::GGSWNoise;
#[cfg(test)]
use poulpy_core::layouts::{Base2K, Dnum, Dsize, Rank, TorusPrecision};
use poulpy_core::layouts::{GGSW, GLWESecretPreparedToRef};
use poulpy_core::{
    LWEFromGLWE, ScratchTakeCore,
    layouts::{GGSWInfos, GGSWPreparedFactory, GLWEInfos, LWE, LWEInfos},
};
#[cfg(test)]
use poulpy_hal::api::ModuleN;
use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

pub(crate) struct FheUintBlocksPreparedDebug<D: Data, T: UnsignedInteger> {
    pub(crate) blocks: Vec<GGSW<D>>,
    pub(crate) _base: u8,
    pub(crate) _phantom: PhantomData<T>,
}

#[cfg(test)]
impl<T: UnsignedInteger> FheUintBlocksPreparedDebug<Vec<u8>, T> {
    #[allow(dead_code)]
    pub(crate) fn alloc<A, M>(module: &M, infos: &A) -> Self
    where
        M: ModuleN,
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
    pub(crate) fn alloc_with<M>(module: &M, base2k: Base2K, k: TorusPrecision, dnum: Dnum, dsize: Dsize, rank: Rank) -> Self
    where
        M: ModuleN,
    {
        Self {
            blocks: (0..T::WORD_SIZE)
                .map(|_| GGSW::alloc(module.n().into(), base2k, k, rank, dnum, dsize))
                .collect(),
            _base: 1,
            _phantom: PhantomData,
        }
    }
}

impl<D: DataRef, T: UnsignedInteger> LWEInfos for FheUintBlocksPreparedDebug<D, T> {
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

impl<D: DataRef, T: UnsignedInteger> GLWEInfos for FheUintBlocksPreparedDebug<D, T> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.blocks[0].rank()
    }
}

impl<D: DataRef, T: UnsignedInteger> GGSWInfos for FheUintBlocksPreparedDebug<D, T> {
    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        self.blocks[0].dsize()
    }

    fn dnum(&self) -> poulpy_core::layouts::Dnum {
        self.blocks[0].dnum()
    }
}

impl<D: DataRef, T: UnsignedInteger + ToBits> FheUintBlocksPreparedDebug<D, T> {
    pub(crate) fn noise<S, M, BE: Backend>(&self, module: &M, sk: &S, want: T)
    where
        S: GLWESecretPreparedToRef<BE>,
        M: GGSWNoise<BE>,
    {
        for (i, ggsw) in self.blocks.iter().enumerate() {
            use poulpy_hal::layouts::{ScalarZnx, ZnxViewMut};
            let mut pt_want = ScalarZnx::alloc(self.n().into(), 1);
            pt_want.at_mut(0, 0)[0] = want.bit(i) as i64;
            ggsw.print_noise(module, sk, &pt_want);
        }
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend, T: UnsignedInteger> FheUintBlockDebugPrepare<BRA, T, BE> for Module<BE>
where
    Self: LWEFromGLWE<BE> + CirtuitBootstrappingExecute<BRA, BE> + GGSWPreparedFactory<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn fhe_uint_block_debug_prepare<DM, DR0, DR1>(
        &self,
        res: &mut FheUintBlocksPreparedDebug<DM, T>,
        bits: &FheUintBlocks<DR0, T>,
        key: &BDDKeyPrepared<DR1, BRA, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DR0: DataRef,
        DR1: DataRef,
    {
        assert_eq!(res.blocks.len(), bits.blocks.len());

        let mut lwe: LWE<Vec<u8>> = LWE::alloc_from_infos(&bits.blocks[0]); //TODO: add TakeLWE
        for (dst, src) in res.blocks.iter_mut().zip(bits.blocks.iter()) {
            lwe.from_glwe(self, src, &key.ks, scratch);
            key.cbt.execute_to_constant(self, dst, &lwe, 1, 1, scratch);
        }
    }
}

impl<D: DataMut, T: UnsignedInteger> FheUintBlocksPreparedDebug<D, T> {
    pub fn prepare<BRA, M, O, K, BE: Backend>(
        &mut self,
        module: &M,
        other: &FheUintBlocks<O, T>,
        key: &BDDKeyPrepared<K, BRA, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        BRA: BlindRotationAlgo,
        O: DataRef,
        K: DataRef,
        M: FheUintBlockDebugPrepare<BRA, T, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.fhe_uint_block_debug_prepare(self, other, key, scratch);
    }
}
