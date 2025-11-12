use crate::tfhe::bdd_arithmetic::FheUintPreparedDebug;
use crate::tfhe::circuit_bootstrapping::CircuitBootstrappingKeyInfos;
use crate::tfhe::{
    bdd_arithmetic::{FheUint, UnsignedInteger},
    blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyFactory},
    circuit_bootstrapping::{
        CircuitBootstrappingKey, CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyLayout,
        CircuitBootstrappingKeyPrepared, CircuitBootstrappingKeyPreparedFactory,
    },
};

use poulpy_core::layouts::{GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyPrepared};
use poulpy_core::{
    GLWEToLWESwitchingKeyEncryptSk, GetDistribution, ScratchTakeCore,
    layouts::{
        GLWEInfos, GLWESecretToRef, GLWEToLWEKey, GLWEToLWEKeyLayout, GLWEToLWEKeyPreparedFactory, LWEInfos, LWESecretToRef,
        prepared::GLWEToLWEKeyPrepared,
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

impl<D: DataRef, BRA: BlindRotationAlgo, BE: Backend> BDDKeyInfos for BDDKeyPrepared<D, BRA, BE> {
    fn cbt_infos(&self) -> CircuitBootstrappingKeyLayout {
        CircuitBootstrappingKeyLayout {
            layout_brk: self.cbt.brk_infos(),
            layout_atk: self.cbt.atk_infos(),
            layout_tsk: self.cbt.tsk_infos(),
        }
    }
    fn ks_infos(&self) -> GLWEToLWEKeyLayout {
        GLWEToLWEKeyLayout {
            n: self.ks.n(),
            base2k: self.ks.base2k(),
            k: self.ks.k(),
            rank_in: self.ks.rank_in(),
            dnum: self.ks.dnum(),
        }
    }
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

impl<D: DataRef, BRA: BlindRotationAlgo, BE: Backend> BDDKeyHelper<D, BRA, BE> for BDDKeyPrepared<D, BRA, BE> {
    fn get_cbt_key(
        &self,
    ) -> (
        &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        &GLWEToLWEKeyPrepared<D, BE>,
    ) {
        (&self.cbt, &self.ks)
    }
}

pub trait BDDKeyHelper<D: DataRef, BRA: BlindRotationAlgo, BE: Backend> {
    fn get_cbt_key(
        &self,
    ) -> (
        &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        &GLWEToLWEKeyPrepared<D, BE>,
    );
}

pub trait FheUintPrepareDebug<BRA: BlindRotationAlgo, T: UnsignedInteger, BE: Backend> {
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
