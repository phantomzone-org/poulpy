use crate::tfhe::bdd_arithmetic::FheUintPreparedDebug;
use crate::tfhe::{
    bdd_arithmetic::{FheUint, FheUintPrepared, UnsignedInteger},
    blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyFactory},
    circuit_bootstrapping::{
        CircuitBootstrappingKey, CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyLayout,
        CircuitBootstrappingKeyPrepared, CircuitBootstrappingKeyPreparedFactory, CirtuitBootstrappingExecute,
    },
};

use poulpy_core::layouts::{GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyPrepared};
use poulpy_core::{
    GLWEToLWESwitchingKeyEncryptSk, GetDistribution, LWEFromGLWE, ScratchTakeCore,
    layouts::{
        GGSWInfos, GGSWPreparedFactory, GLWEInfos, GLWESecretToRef, GLWEToLWEKey, GLWEToLWEKeyLayout,
        GLWEToLWEKeyPreparedFactory, LWE, LWEInfos, LWESecretToRef, prepared::GLWEToLWEKeyPrepared,
    },
};
use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
    source::Source,
};

pub trait BDDKeyInfos {
    fn cbt_infos(&self) -> CircuitBootstrappingKeyLayout;
    fn ks_infos(&self) -> GLWEToLWEKeyLayout;
}

#[derive(Debug, Clone, Copy)]
pub struct BDDKeyLayout {
    pub cbt: CircuitBootstrappingKeyLayout,
    pub ks: GLWEToLWEKeyLayout,
}

impl BDDKeyInfos for BDDKeyLayout {
    fn cbt_infos(&self) -> CircuitBootstrappingKeyLayout {
        self.cbt
    }

    fn ks_infos(&self) -> GLWEToLWEKeyLayout {
        self.ks
    }
}

pub struct BDDKey<D, BRA>
where
    D: Data,
    BRA: BlindRotationAlgo,
{
    pub(crate) cbt: CircuitBootstrappingKey<D, BRA>,
    pub(crate) ks: GLWEToLWEKey<D>,
}

impl<BRA: BlindRotationAlgo> BDDKey<Vec<u8>, BRA>
where
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>,
{
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: BDDKeyInfos,
    {
        Self {
            cbt: CircuitBootstrappingKey::alloc_from_infos(&infos.cbt_infos()),
            ks: GLWEToLWEKey::alloc_from_infos(&infos.ks_infos()),
        }
    }
}

pub trait BDDKeyEncryptSk<BRA: BlindRotationAlgo, BE: Backend> {
    fn bdd_key_encrypt_sk<D, S0, S1>(
        &self,
        res: &mut BDDKey<D, BRA>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        D: DataMut,
        S0: LWESecretToRef + GetDistribution + LWEInfos,
        S1: GLWESecretToRef + GetDistribution + GLWEInfos;
}

impl<BE: Backend, BRA: BlindRotationAlgo> BDDKeyEncryptSk<BRA, BE> for Module<BE>
where
    Self: CircuitBootstrappingKeyEncryptSk<BRA, BE> + GLWEToLWESwitchingKeyEncryptSk<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn bdd_key_encrypt_sk<D, S0, S1>(
        &self,
        res: &mut BDDKey<D, BRA>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        D: DataMut,
        S0: LWESecretToRef + GetDistribution + LWEInfos,
        S1: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        res.ks
            .encrypt_sk(self, sk_lwe, sk_glwe, source_xa, source_xe, scratch);
        res.cbt
            .encrypt_sk(self, sk_lwe, sk_glwe, source_xa, source_xe, scratch);
    }
}

impl<D: DataMut, BRA: BlindRotationAlgo> BDDKey<D, BRA> {
    pub fn encrypt_sk<S0, S1, M, BE: Backend>(
        &mut self,
        module: &M,
        sk_lwe: &S0,
        sk_glwe: &S1,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S0: LWESecretToRef + GetDistribution + LWEInfos,
        S1: GLWESecretToRef + GetDistribution + GLWEInfos,
        M: BDDKeyEncryptSk<BRA, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.bdd_key_encrypt_sk(self, sk_lwe, sk_glwe, source_xa, source_xe, scratch);
    }
}

pub struct BDDKeyPrepared<D, BRA, BE>
where
    D: Data,
    BRA: BlindRotationAlgo,
    BE: Backend,
{
    pub(crate) cbt: CircuitBootstrappingKeyPrepared<D, BRA, BE>,
    pub(crate) ks: GLWEToLWEKeyPrepared<D, BE>,
}

impl<D: DataRef, BRA: BlindRotationAlgo, BE: Backend> GLWEAutomorphismKeyHelper<GLWEAutomorphismKeyPrepared<D, BE>, BE>
    for BDDKeyPrepared<D, BRA, BE>
{
    fn automorphism_key_infos(&self) -> poulpy_core::layouts::GGLWELayout {
        self.cbt.automorphism_key_infos()
    }

    fn get_automorphism_key(&self, k: i64) -> Option<&GLWEAutomorphismKeyPrepared<D, BE>> {
        self.cbt.get_automorphism_key(k)
    }
}

pub trait BDDKeyPreparedFactory<BRA: BlindRotationAlgo, BE: Backend>
where
    Self: Sized + CircuitBootstrappingKeyPreparedFactory<BRA, BE> + GLWEToLWEKeyPreparedFactory<BE>,
{
    fn alloc_bdd_key_from_infos<A>(&self, infos: &A) -> BDDKeyPrepared<Vec<u8>, BRA, BE>
    where
        A: BDDKeyInfos,
    {
        BDDKeyPrepared {
            cbt: CircuitBootstrappingKeyPrepared::alloc_from_infos(self, &infos.cbt_infos()),
            ks: GLWEToLWEKeyPrepared::alloc_from_infos(self, &infos.ks_infos()),
        }
    }

    fn prepare_bdd_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos,
    {
        self.circuit_bootstrapping_key_prepare_tmp_bytes(&infos.cbt_infos())
            .max(self.prepare_glwe_to_lwe_key_tmp_bytes(&infos.ks_infos()))
    }

    fn prepare_bdd_key<DM, DR>(&self, res: &mut BDDKeyPrepared<DM, BRA, BE>, other: &BDDKey<DR, BRA>, scratch: &mut Scratch<BE>)
    where
        DM: DataMut,
        DR: DataRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        res.cbt.prepare(self, &other.cbt, scratch);
        res.ks.prepare(self, &other.ks, scratch);
    }
}
impl<BRA: BlindRotationAlgo, BE: Backend> BDDKeyPreparedFactory<BRA, BE> for Module<BE> where
    Self: Sized + CircuitBootstrappingKeyPreparedFactory<BRA, BE> + GLWEToLWEKeyPreparedFactory<BE>
{
}

impl<BRA: BlindRotationAlgo, BE: Backend> BDDKeyPrepared<Vec<u8>, BRA, BE> {
    pub fn alloc_from_infos<M, A>(module: &M, infos: &A) -> Self
    where
        M: BDDKeyPreparedFactory<BRA, BE>,
        A: BDDKeyInfos,
    {
        module.alloc_bdd_key_from_infos(infos)
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

pub trait FheUintBlocksPrepare<BRA: BlindRotationAlgo, T: UnsignedInteger, BE: Backend> {
    fn fhe_uint_prepare_tmp_bytes<R, A>(&self, block_size: usize, extension_factor: usize, res_infos: &R, infos: &A) -> usize
    where
        R: GGSWInfos,
        A: BDDKeyInfos;
    fn fhe_uint_prepare<DM, DR0, DR1>(
        &self,
        res: &mut FheUintPrepared<DM, T, BE>,
        bits: &FheUint<DR0, T>,
        key: &BDDKeyPrepared<DR1, BRA, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DR0: DataRef,
        DR1: DataRef;
}

impl<BRA: BlindRotationAlgo, BE: Backend, T: UnsignedInteger> FheUintBlocksPrepare<BRA, T, BE> for Module<BE>
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

    fn fhe_uint_prepare<DM, DR0, DR1>(
        &self,
        res: &mut FheUintPrepared<DM, T, BE>,
        bits: &FheUint<DR0, T>,
        key: &BDDKeyPrepared<DR1, BRA, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DR0: DataRef,
        DR1: DataRef,
    {
        let mut lwe: LWE<Vec<u8>> = LWE::alloc_from_infos(bits); //TODO: add TakeLWE
        let (mut tmp_ggsw, scratch_1) = scratch.take_ggsw(res);
        for (bit, dst) in res.bits.iter_mut().enumerate() {
            bits.get_bit(self, bit, &mut lwe, &key.ks, scratch_1);
            key.cbt
                .execute_to_constant(self, &mut tmp_ggsw, &lwe, 1, 1, scratch_1);
            dst.prepare(self, &tmp_ggsw, scratch_1);
        }
    }
}

impl<D: DataMut, T: UnsignedInteger, BE: Backend> FheUintPrepared<D, T, BE> {
    pub fn prepare<BRA, M, O, K>(
        &mut self,
        module: &M,
        other: &FheUint<O, T>,
        key: &BDDKeyPrepared<K, BRA, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        BRA: BlindRotationAlgo,
        O: DataRef,
        K: DataRef,
        M: FheUintBlocksPrepare<BRA, T, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.fhe_uint_prepare(self, other, key, scratch);
    }
}

pub trait FheUintBlockDebugPrepare<BRA: BlindRotationAlgo, T: UnsignedInteger, BE: Backend> {
    fn fhe_uint_debug_prepare<DM, DR0, DR1>(
        &self,
        res: &mut FheUintPreparedDebug<DM, T>,
        bits: &FheUint<DR0, T>,
        key: &BDDKeyPrepared<DR1, BRA, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DR0: DataRef,
        DR1: DataRef;
}
