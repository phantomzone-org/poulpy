use std::marker::PhantomData;
use std::thread;

use poulpy_core::layouts::{
    Base2K, Dnum, Dsize, GGSWInfos, GGSWPreparedFactory, GLWEInfos, LWEInfos, Rank, TorusPrecision, prepared::GGSWPrepared,
};
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToRef, GGSW, GGSWLayout, GGSWPreparedToMut, GGSWPreparedToRef, GLWEAutomorphismKeyHelper,
    GetGaloisElement, LWE,
};
use poulpy_core::{GLWECopy, GLWEDecrypt, GLWEPacking, LWEFromGLWE};

use poulpy_core::{GGSWEncryptSk, ScratchTakeCore, layouts::GLWESecretPreparedToRef};
use poulpy_hal::api::{ModuleLogN, ScratchAvailable, ScratchFromBytes};
use poulpy_hal::layouts::{Backend, Data, DataRef, Module};

use poulpy_hal::{
    api::ModuleN,
    layouts::{DataMut, Scratch},
    source::Source,
};

use crate::tfhe::bdd_arithmetic::{BDDKey, BDDKeyHelper, BDDKeyInfos, BDDKeyPrepared, BDDKeyPreparedFactory, FheUint, ToBits};
use crate::tfhe::bdd_arithmetic::{Cmux, FromBits, ScratchTakeBDD, UnsignedInteger};
use crate::tfhe::blind_rotation::BlindRotationAlgo;
use crate::tfhe::circuit_bootstrapping::{CircuitBootstrappingKeyInfos, CirtuitBootstrappingExecute};

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

impl<D: DataRef, T: UnsignedInteger + FromBits, BE: Backend> FheUintPrepared<D, T, BE> {
    pub fn decrypt<M, S, H, K>(&self, module: &M, sk: &S, keys: &H, scratch: &mut Scratch<BE>) -> T
    where
        M: ModuleLogN + GLWEDecrypt<BE> + Cmux<BE> + GLWEPacking<BE> + GLWECopy,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let (mut tmp, scratch_1) = scratch.take_fhe_uint(self);
        tmp.from_fhe_uint_prepared(module, self, keys, scratch_1);
        tmp.decrypt(module, sk, scratch_1)
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

pub trait FheUintPrepare<BRA: BlindRotationAlgo, BE: Backend> {
    fn fhe_uint_prepare_tmp_bytes<R, A, B>(&self, block_size: usize, extension_factor: usize, res_infos: &R, bits_infos: &A, bdd_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GLWEInfos,
        B: BDDKeyInfos;
    fn fhe_uint_prepare<DM, DB, DK, K, T: UnsignedInteger>(
        &self,
        res: &mut FheUintPrepared<DM, T, BE>,
        bits: &FheUint<DB, T>,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DB: DataRef,
        DK: DataRef,
        K: BDDKeyHelper<DK, BRA, BE> + BDDKeyInfos,
        Scratch<BE>: ScratchFromBytes<BE>,
    {
        self.fhe_uint_prepare_custom(res, bits, 0, T::BITS as usize, key, scratch);
    }
    fn fhe_uint_prepare_custom<DM, DB, DK, K, T: UnsignedInteger>(
        &self,
        res: &mut FheUintPrepared<DM, T, BE>,
        bits: &FheUint<DB, T>,
        bit_start: usize,
        bit_count: usize,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DB: DataRef,
        DK: DataRef,
        K: BDDKeyHelper<DK, BRA, BE> + BDDKeyInfos,
    {
        self.fhe_uint_prepare_custom_multi_thread(1, res, bits, bit_start, bit_count, key, scratch)
    }
    fn fhe_uint_prepare_custom_multi_thread<DM, DB, DK, K, T: UnsignedInteger>(
        &self,
        threads: usize,
        res: &mut FheUintPrepared<DM, T, BE>,
        bits: &FheUint<DB, T>,
        bit_start: usize,
        bit_count: usize,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DB: DataRef,
        DK: DataRef,
        K: BDDKeyHelper<DK, BRA, BE> + BDDKeyInfos;
}

impl<BRA: BlindRotationAlgo, BE: Backend> FheUintPrepare<BRA, BE> for Module<BE>
where
    Self: LWEFromGLWE<BE> + CirtuitBootstrappingExecute<BRA, BE> + GGSWPreparedFactory<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn fhe_uint_prepare_tmp_bytes<R, A, B>(&self, block_size: usize, extension_factor: usize, res_infos: &R, bits_infos: &A, bdd_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GLWEInfos,
        B: BDDKeyInfos,
    {
        self.circuit_bootstrapping_execute_tmp_bytes(
            block_size,
            extension_factor,
            res_infos,
            &bdd_infos.cbt_infos(),
        ) + GGSW::bytes_of_from_infos(res_infos)
            + LWE::bytes_of_from_infos(bits_infos)
    }

    fn fhe_uint_prepare_custom_multi_thread<DM, DB, DK, K, T: UnsignedInteger>(
        &self,
        threads: usize,
        res: &mut FheUintPrepared<DM, T, BE>,
        bits: &FheUint<DB, T>,
        bit_start: usize,
        bit_count: usize,
        key: &K,
        mut scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DB: DataRef,
        DK: DataRef,
        K: BDDKeyHelper<DK, BRA, BE> + BDDKeyInfos,
    {
        let bit_end = bit_start + bit_count;
        let (cbt, ks) = key.get_cbt_key();

        assert!(bit_end <= T::BITS as usize);

        let scratch_thread_size = self.fhe_uint_prepare_tmp_bytes(cbt.block_size(), 1, res, bits, key);

        assert!(scratch.available() >= threads * scratch_thread_size);

        // How many bits we need to process
        let chunk_size: usize = bit_count.div_ceil(threads); // ceil division

        let mut scratches = Vec::new();
        for _ in 0..(threads - 1) {
            let (tmp, scratch_new) = scratch.split_at_mut(scratch_thread_size);
            scratch = scratch_new;
            scratches.push(tmp);
        }
        scratches.push(scratch);

        let ggsw_infos: &GGSWLayout = &res.ggsw_layout();

        thread::scope(|scope| {
            for (thread_index, (scratch_thread, res_bits_chunk)) in scratches
                .iter_mut()
                .zip(res.bits[bit_start..bit_end].chunks_mut(chunk_size))
                .enumerate()
            {
                let start: usize = bit_start + thread_index * chunk_size;

                scope.spawn(move || {
                    let (mut tmp_ggsw, scratch_1) = scratch_thread.take_ggsw(ggsw_infos);
                    let (mut tmp_lwe, scratch_2) = scratch_1.take_lwe(bits);
                    for (local_bit, dst) in res_bits_chunk.iter_mut().enumerate() {
                        bits.get_bit_lwe(self, start + local_bit, &mut tmp_lwe, ks, scratch_2);
                        cbt.execute_to_constant(self, &mut tmp_ggsw, &tmp_lwe, 1, 1, scratch_2);
                        dst.prepare(self, &tmp_ggsw, scratch_2);
                    }
                });
            }
        });

        for i in 0..bit_start {
            res.bits[i].zero(self);
        }

        for i in bit_end..T::BITS as usize {
            res.bits[i].zero(self);
        }
    }
}

impl<D: DataMut, T: UnsignedInteger, BE: Backend> FheUintPrepared<D, T, BE> {
    pub fn prepare<BRA, M, O, K, DK>(&mut self, module: &M, other: &FheUint<O, T>, key: &K, scratch: &mut Scratch<BE>)
    where
        BRA: BlindRotationAlgo,
        O: DataRef,
        DK: DataRef,
        K: BDDKeyHelper<DK, BRA, BE> + BDDKeyInfos,
        M: FheUintPrepare<BRA, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.fhe_uint_prepare(self, other, key, scratch);
    }
    pub fn prepare_custom<BRA, M, O, K, DK>(
        &mut self,
        module: &M,
        other: &FheUint<O, T>,
        bit_start: usize,
        bit_end: usize,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) where
        BRA: BlindRotationAlgo,
        O: DataRef,
        DK: DataRef,
        K: BDDKeyHelper<DK, BRA, BE> + BDDKeyInfos,
        M: FheUintPrepare<BRA, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.fhe_uint_prepare_custom(self, other, bit_start, bit_end, key, scratch);
    }

    pub fn prepare_custom_multi_thread<BRA, M, O, K, DK>(
        &mut self,
        threads: usize,
        module: &M,
        other: &FheUint<O, T>,
        bit_start: usize,
        bit_end: usize,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) where
        BRA: BlindRotationAlgo,
        O: DataRef,
        DK: DataRef,
        K: BDDKeyHelper<DK, BRA, BE> + BDDKeyInfos,
        M: FheUintPrepare<BRA, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.fhe_uint_prepare_custom_multi_thread(threads, self, other, bit_start, bit_end, key, scratch);
    }
}
