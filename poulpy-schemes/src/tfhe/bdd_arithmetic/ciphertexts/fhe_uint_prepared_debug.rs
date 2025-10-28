use std::marker::PhantomData;

use crate::tfhe::bdd_arithmetic::{BDDKeyPrepared, FheUint, FheUintBlockDebugPrepare, ToBits};
use crate::tfhe::{
    bdd_arithmetic::UnsignedInteger, blind_rotation::BlindRotationAlgo, circuit_bootstrapping::CirtuitBootstrappingExecute,
};
use poulpy_core::GGSWNoise;

use poulpy_core::layouts::{Base2K, Dnum, Dsize, Rank, TorusPrecision};
use poulpy_core::layouts::{GGSW, GLWESecretPreparedToRef};
use poulpy_core::{
    LWEFromGLWE, ScratchTakeCore,
    layouts::{GGSWInfos, GGSWPreparedFactory, GLWEInfos, LWE, LWEInfos},
};

use poulpy_hal::api::ModuleN;
use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

pub struct FheUintPreparedDebug<D: Data, T: UnsignedInteger> {
    pub(crate) bits: Vec<GGSW<D>>,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T: UnsignedInteger> FheUintPreparedDebug<Vec<u8>, T> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        M: ModuleN,
        A: GGSWInfos,
    {
        Self::alloc(
            module,
            infos.base2k(),
            infos.k(),
            infos.dnum(),
            infos.dsize(),
            infos.rank(),
        )
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, dnum: Dnum, dsize: Dsize, rank: Rank) -> Self
    where
        M: ModuleN,
    {
        Self {
            bits: (0..T::WORD_SIZE)
                .map(|_| GGSW::alloc(module.n().into(), base2k, k, rank, dnum, dsize))
                .collect(),
            _phantom: PhantomData,
        }
    }
}

impl<D: DataRef, T: UnsignedInteger> LWEInfos for FheUintPreparedDebug<D, T> {
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.bits[0].base2k()
    }

    fn k(&self) -> poulpy_core::layouts::TorusPrecision {
        self.bits[0].k()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.bits[0].n()
    }
}

impl<D: DataRef, T: UnsignedInteger> GLWEInfos for FheUintPreparedDebug<D, T> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.bits[0].rank()
    }
}

impl<D: DataRef, T: UnsignedInteger> GGSWInfos for FheUintPreparedDebug<D, T> {
    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        self.bits[0].dsize()
    }

    fn dnum(&self) -> poulpy_core::layouts::Dnum {
        self.bits[0].dnum()
    }
}

impl<D: DataRef, T: UnsignedInteger + ToBits> FheUintPreparedDebug<D, T> {
    pub fn print_noise<S, M, BE: Backend>(&self, module: &M, sk: &S, want: T)
    where
        S: GLWESecretPreparedToRef<BE>,
        M: GGSWNoise<BE>,
    {
        for (i, ggsw) in self.bits.iter().enumerate() {
            use poulpy_hal::layouts::{ScalarZnx, ZnxViewMut};
            let mut pt_want = ScalarZnx::alloc(self.n().into(), 1);
            pt_want.at_mut(0, 0)[0] = want.bit(i) as i64;
            ggsw.print_noise(module, sk, &pt_want);
        }
    }

    pub fn assert_noise<S, M, F, BE: Backend>(&self, module: &M, sk: &S, want: T, max_noise: &F)
    where
        S: GLWESecretPreparedToRef<BE>,
        M: GGSWNoise<BE>,
        F: Fn(usize) -> f64,
    {
        for (i, ggsw) in self.bits.iter().enumerate() {
            use poulpy_hal::layouts::{ScalarZnx, ZnxViewMut};
            let mut pt_want = ScalarZnx::alloc(self.n().into(), 1);
            pt_want.at_mut(0, 0)[0] = want.bit(i) as i64;
            ggsw.assert_noise(module, sk, &pt_want, max_noise);
        }
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend, T: UnsignedInteger> FheUintBlockDebugPrepare<BRA, T, BE> for Module<BE>
where
    Self: ModuleN + LWEFromGLWE<BE> + CirtuitBootstrappingExecute<BRA, BE> + GGSWPreparedFactory<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn fhe_uint_debug_prepare<DM, DR0, DR1>(
        &self,
        res: &mut FheUintPreparedDebug<DM, T>,
        bits: &FheUint<DR0, T>,
        key: &BDDKeyPrepared<DR1, BRA, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DR0: DataRef,
        DR1: DataRef,
    {
        let mut lwe: LWE<Vec<u8>> = LWE::alloc_from_infos(bits); //TODO: add TakeLWE
        for (bit, dst) in res.bits.iter_mut().enumerate() {
            bits.get_bit(self, bit, &mut lwe, &key.ks, scratch);
            key.cbt.execute_to_constant(self, dst, &lwe, 1, 1, scratch);
        }
    }
}

impl<D: DataMut, T: UnsignedInteger> FheUintPreparedDebug<D, T> {
    pub fn prepare<BRA, M, O, K, BE: Backend>(
        &mut self,
        module: &M,
        other: &FheUint<O, T>,
        key: &BDDKeyPrepared<K, BRA, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        BRA: BlindRotationAlgo,
        O: DataRef,
        K: DataRef,
        M: FheUintBlockDebugPrepare<BRA, T, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.fhe_uint_debug_prepare(self, other, key, scratch);
    }
}
