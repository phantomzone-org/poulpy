use std::marker::PhantomData;

use poulpy_core::layouts::{
    Base2K, Dnum, Dsize, GGSWInfos, GGSWPreparedFactory, GLWEInfos, LWEInfos, Rank, TorusPrecision, prepared::GGSWPrepared,
};
use poulpy_core::layouts::{GGSWPreparedToMut, GGSWPreparedToRef};

use poulpy_core::{GGSWEncryptSk, ScratchTakeCore, layouts::GLWESecretPreparedToRef};
use poulpy_hal::layouts::{Backend, Data, DataRef, Module};

use poulpy_hal::{
    api::ModuleN,
    layouts::{DataMut, Scratch},
    source::Source,
};

use crate::tfhe::bdd_arithmetic::ToBits;
use crate::tfhe::bdd_arithmetic::UnsignedInteger;

/// A prepared FHE ciphertext encrypting the bits of an [UnsignedInteger].
pub struct FheUintPrepared<D: Data, T: UnsignedInteger, B: Backend> {
    pub(crate) bits: Vec<GGSWPrepared<D, B>>,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T: UnsignedInteger, BE: Backend> FheUintBlocksPreparedFactory<T, BE> for Module<BE> where
    Self: Sized + GGSWPreparedFactory<BE>
{
}

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

pub trait FheUintBlocksPreparedFactory<T: UnsignedInteger, BE: Backend>
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
            bits: (0..T::WORD_SIZE)
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
    pub fn alloc<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGSWInfos,
        M: FheUintBlocksPreparedFactory<T, BE>,
    {
        module.alloc_fhe_uint_prepared_from_infos(infos)
    }

    pub fn alloc_with<M>(module: &M, base2k: Base2K, k: TorusPrecision, dnum: Dnum, dsize: Dsize, rank: Rank) -> Self
    where
        M: FheUintBlocksPreparedFactory<T, BE>,
    {
        module.alloc_fhe_uint_prepared(base2k, k, dnum, dsize, rank)
    }
}

impl<T: UnsignedInteger + ToBits, BE: Backend> FheUintBlocksPreparedEncryptSk<T, BE> for Module<BE> where
    Self: Sized + ModuleN + GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>
{
}

pub trait FheUintBlocksPreparedEncryptSk<T: UnsignedInteger + ToBits, BE: Backend>
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

        assert!(self.n().is_multiple_of(T::WORD_SIZE));
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk.n(), self.n() as u32);

        let (mut tmp_ggsw, scratch_1) = scratch.take_ggsw(res);
        let (mut pt, scratch_2) = scratch_1.take_scalar_znx(self.n(), 1);
        pt.zero();

        for i in 0..T::WORD_SIZE {
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
        M: FheUintBlocksPreparedEncryptSk<T, BE>,
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
