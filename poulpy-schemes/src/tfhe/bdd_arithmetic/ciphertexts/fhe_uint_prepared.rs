use std::marker::PhantomData;

use poulpy_core::LWEFromGLWE;
use poulpy_core::layouts::{
    Base2K, Dnum, Dsize, GGSWInfos, GGSWPreparedFactory, GLWEInfos, LWEInfos, Rank, TorusPrecision, prepared::GGSWPrepared,
};
use poulpy_core::layouts::{GGSWPreparedToMut, GGSWPreparedToRef, LWE};

use poulpy_core::{GGSWEncryptSk, ScratchTakeCore, layouts::GLWESecretPreparedToRef};
use poulpy_hal::layouts::{Backend, Data, DataRef, Module};

use poulpy_hal::{
    api::ModuleN,
    layouts::{DataMut, Scratch},
    source::Source,
};

use crate::tfhe::bdd_arithmetic::UnsignedInteger;
use crate::tfhe::bdd_arithmetic::{BDDKey, BDDKeyHelper, BDDKeyInfos, BDDKeyPrepared, BDDKeyPreparedFactory, FheUint, ToBits};
use crate::tfhe::blind_rotation::BlindRotationAlgo;
use crate::tfhe::circuit_bootstrapping::CirtuitBootstrappingExecute;

/// A prepared FHE ciphertext encrypting the bits of an [UnsignedInteger].
pub struct FheUintPrepared<D: Data, T: UnsignedInteger, B: Backend> {
    pub(crate) bits: Vec<GGSWPrepared<D, B>>,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T: UnsignedInteger, BE: Backend> FheUintPreparedFactory<T, BE> for Module<BE> where Self: Sized + GGSWPreparedFactory<BE> {}

pub trait GetGGSWBit<BE: Backend> {
    fn get_bit(&self, bit: usize) -> GGSWPrepared<&[u8], BE>;
}

impl<D: DataRef, T: UnsignedInteger, BE: Backend> GetGGSWBit<BE> for FheUintPrepared<D, T, BE> {
    fn get_bit(&self, bit: usize) -> GGSWPrepared<&[u8], BE> {
        assert!(bit <= self.bits.len());
        self.bits[bit].to_ref()
    }
}

pub trait GetGGSWBitMut<T: UnsignedInteger, BE: Backend> {
    fn get_bit(&mut self, bit: usize) -> GGSWPrepared<&mut [u8], BE>;
}

impl<D: DataMut, T: UnsignedInteger, BE: Backend> GetGGSWBitMut<T, BE> for FheUintPrepared<D, T, BE> {
    fn get_bit(&mut self, bit: usize) -> GGSWPrepared<&mut [u8], BE> {
        assert!(bit <= self.bits.len());
        self.bits[bit].to_mut()
    }
}

pub trait FheUintPreparedFactory<T: UnsignedInteger, BE: Backend>
where
    Self: Sized + GGSWPreparedFactory<BE>,
{
    fn alloc_fhe_uint_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        dnum: Dnum,
        dsize: Dsize,
        rank: Rank,
    ) -> FheUintPrepared<Vec<u8>, T, BE> {
        FheUintPrepared {
            bits: (0..T::BITS)
                .map(|_| GGSWPrepared::alloc(self, base2k, k, dnum, dsize, rank))
                .collect(),
            _phantom: PhantomData,
        }
    }

    fn alloc_fhe_uint_prepared_from_infos<A>(&self, infos: &A) -> FheUintPrepared<Vec<u8>, T, BE>
    where
        A: GGSWInfos,
    {
        self.alloc_fhe_uint_prepared(
            infos.base2k(),
            infos.k(),
            infos.dnum(),
            infos.dsize(),
            infos.rank(),
        )
    }
}

impl<T: UnsignedInteger, BE: Backend> FheUintPrepared<Vec<u8>, T, BE> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGSWInfos,
        M: FheUintPreparedFactory<T, BE>,
    {
        module.alloc_fhe_uint_prepared_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, dnum: Dnum, dsize: Dsize, rank: Rank) -> Self
    where
        M: FheUintPreparedFactory<T, BE>,
    {
        module.alloc_fhe_uint_prepared(base2k, k, dnum, dsize, rank)
    }
}

impl<T: UnsignedInteger + ToBits, BE: Backend> FheUintPreparedEncryptSk<T, BE> for Module<BE> where
    Self: Sized + ModuleN + GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>
{
}

pub trait FheUintPreparedEncryptSk<T: UnsignedInteger + ToBits, BE: Backend>
where
    Self: Sized + ModuleN + GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>,
{
    fn fhe_uint_prepared_encrypt_sk<DM, S>(
        &self,
        res: &mut FheUintPrepared<DM, T, BE>,
        value: T,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        use poulpy_hal::{api::ScratchTakeBasic, layouts::ZnxZero};

        assert!(self.n().is_multiple_of(T::BITS as usize));
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk.n(), self.n() as u32);

        let (mut tmp_ggsw, scratch_1) = scratch.take_ggsw(res);
        let (mut pt, scratch_2) = scratch_1.take_scalar_znx(self.n(), 1);
        pt.zero();

        for i in 0..T::BITS as usize {
            use poulpy_hal::layouts::ZnxViewMut;
            pt.at_mut(0, 0)[0] = value.bit(i) as i64;
            tmp_ggsw.encrypt_sk(self, &pt, sk, source_xa, source_xe, scratch_2);
            res.bits[i].prepare(self, &tmp_ggsw, scratch_2);
        }
    }
}

impl<D: DataMut, T: UnsignedInteger + ToBits, BE: Backend> FheUintPrepared<D, T, BE> {
    pub fn encrypt_sk<M, S>(
        &mut self,
        module: &M,
        value: T,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        M: FheUintPreparedEncryptSk<T, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.fhe_uint_prepared_encrypt_sk(self, value, sk, source_xa, source_xe, scratch);
    }
}

impl<D: DataRef, T: UnsignedInteger, B: Backend> LWEInfos for FheUintPrepared<D, T, B> {
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

impl<D: DataRef, T: UnsignedInteger, B: Backend> GLWEInfos for FheUintPrepared<D, T, B> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.bits[0].rank()
    }
}

impl<D: DataRef, T: UnsignedInteger, B: Backend> GGSWInfos for FheUintPrepared<D, T, B> {
    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        self.bits[0].dsize()
    }

    fn dnum(&self) -> poulpy_core::layouts::Dnum {
        self.bits[0].dnum()
    }
}

impl<D: DataMut, BRA: BlindRotationAlgo, BE: Backend> BDDKeyPrepared<D, BRA, BE> {
    pub fn prepare<DR, M>(&mut self, module: &M, other: &BDDKey<DR, BRA>, scratch: &mut Scratch<BE>)
    where
        DR: DataRef,
        M: BDDKeyPreparedFactory<BRA, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.prepare_bdd_key(self, other, scratch);
    }
}

pub trait FheUintPrepare<BRA: BlindRotationAlgo, T: UnsignedInteger, BE: Backend> {
    fn fhe_uint_prepare_tmp_bytes<R, A>(&self, block_size: usize, extension_factor: usize, res_infos: &R, infos: &A) -> usize
    where
        R: GGSWInfos,
        A: BDDKeyInfos;
    fn fhe_uint_prepare<DM, DB, DK, K>(
        &self,
        res: &mut FheUintPrepared<DM, T, BE>,
        bits: &FheUint<DB, T>,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DB: DataRef,
        DK: DataRef,
        K: BDDKeyHelper<DK, BRA, BE>;
    fn fhe_uint_prepare_partial<DM, DB, DK, K>(
        &self,
        res: &mut FheUintPrepared<DM, T, BE>,
        bits: &FheUint<DB, T>,
        count: usize,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DB: DataRef,
        DK: DataRef,
        K: BDDKeyHelper<DK, BRA, BE>;        
}

impl<BRA: BlindRotationAlgo, BE: Backend, T: UnsignedInteger> FheUintPrepare<BRA, T, BE> for Module<BE>
where
    Self: LWEFromGLWE<BE> + CirtuitBootstrappingExecute<BRA, BE> + GGSWPreparedFactory<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn fhe_uint_prepare_tmp_bytes<R, A>(&self, block_size: usize, extension_factor: usize, res_infos: &R, bdd_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: BDDKeyInfos,
    {
        self.circuit_bootstrapping_execute_tmp_bytes(
            block_size,
            extension_factor,
            res_infos,
            &bdd_infos.cbt_infos(),
        )
    }

    fn fhe_uint_prepare<DM, DB, DK, K>(
        &self,
        res: &mut FheUintPrepared<DM, T, BE>,
        bits: &FheUint<DB, T>,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DB: DataRef,
        DK: DataRef,
        K: BDDKeyHelper<DK, BRA, BE>,
    {
        let (cbt, ks) = key.get_cbt_key();

        let mut lwe: LWE<Vec<u8>> = LWE::alloc_from_infos(bits); //TODO: add TakeLWE
        let (mut tmp_ggsw, scratch_1) = scratch.take_ggsw(res);
        for (bit, dst) in res.bits.iter_mut().enumerate() {
            bits.get_bit_lwe(self, bit, &mut lwe, ks, scratch_1);
            cbt.execute_to_constant(self, &mut tmp_ggsw, &lwe, 1, 1, scratch_1);
            dst.prepare(self, &tmp_ggsw, scratch_1);
        }
    }

    fn fhe_uint_prepare_partial<DM, DB, DK, K>(
        &self,
        res: &mut FheUintPrepared<DM, T, BE>,
        bits: &FheUint<DB, T>,
        count: usize,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DB: DataRef,
        DK: DataRef,
        K: BDDKeyHelper<DK, BRA, BE>,
    {
        let (cbt, ks) = key.get_cbt_key();

        let mut lwe: LWE<Vec<u8>> = LWE::alloc_from_infos(bits); //TODO: add TakeLWE
        let (mut tmp_ggsw, scratch_1) = scratch.take_ggsw(res);
        for (bit, dst) in res.bits[0..count].iter_mut().enumerate() { // TODO: set the rest of the bits to a prepared zero GGSW
            bits.get_bit_lwe(self, bit, &mut lwe, ks, scratch_1);
            cbt.execute_to_constant(self, &mut tmp_ggsw, &lwe, 1, 1, scratch_1);
            dst.prepare(self, &tmp_ggsw, scratch_1);
        }
    }    
}

impl<D: DataMut, T: UnsignedInteger, BE: Backend> FheUintPrepared<D, T, BE> {
    pub fn prepare<BRA, M, O, K, DK>(&mut self, module: &M, other: &FheUint<O, T>, key: &K, scratch: &mut Scratch<BE>)
    where
        BRA: BlindRotationAlgo,
        O: DataRef,
        DK: DataRef,
        K: BDDKeyHelper<DK, BRA, BE>,
        M: FheUintPrepare<BRA, T, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.fhe_uint_prepare(self, other, key, scratch);
    }
    pub fn prepare_partial<BRA, M, O, K, DK>(&mut self, module: &M, other: &FheUint<O, T>, count: usize, key: &K, scratch: &mut Scratch<BE>)
    where
        BRA: BlindRotationAlgo,
        O: DataRef,
        DK: DataRef,
        K: BDDKeyHelper<DK, BRA, BE>,
        M: FheUintPrepare<BRA, T, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.fhe_uint_prepare_partial(self, other, count, key, scratch);
    }    
}
