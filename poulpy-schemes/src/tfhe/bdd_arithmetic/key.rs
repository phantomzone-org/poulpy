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

use poulpy_core::GLWESwitchingKeyEncryptSk;
use poulpy_core::layouts::{
    GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyPrepared, GLWESecret, GLWESwitchingKey, GLWESwitchingKeyLayout,
    GLWESwitchingKeyPrepared,
};
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
    fn ks_lwe_infos(&self) -> GLWEToLWEKeyLayout;
    fn ks_glwe_infos(&self) -> Option<GLWESwitchingKeyLayout>;
}

#[derive(Debug, Clone, Copy)]
pub struct BDDKeyLayout {
    pub cbt: CircuitBootstrappingKeyLayout,
    pub ks_glwe: Option<GLWESwitchingKeyLayout>,
    pub ks_lwe: GLWEToLWEKeyLayout,
}

impl BDDKeyInfos for BDDKeyLayout {
    fn cbt_infos(&self) -> CircuitBootstrappingKeyLayout {
        self.cbt
    }

    fn ks_glwe_infos(&self) -> Option<GLWESwitchingKeyLayout> {
        self.ks_glwe
    }

    fn ks_lwe_infos(&self) -> GLWEToLWEKeyLayout {
        self.ks_lwe
    }
}

pub struct BDDKey<D, BRA>
where
    D: Data,
    BRA: BlindRotationAlgo,
{
    pub(crate) cbt: CircuitBootstrappingKey<D, BRA>,
    pub(crate) ks_glwe: Option<GLWESwitchingKey<D>>,
    pub(crate) ks_lwe: GLWEToLWEKey<D>,
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
            ks_glwe: infos
                .ks_glwe_infos()
                .as_ref()
                .map(GLWESwitchingKey::alloc_from_infos),
            ks_lwe: GLWEToLWEKey::alloc_from_infos(&infos.ks_lwe_infos()),
        }
    }
}

pub trait BDDKeyEncryptSk<BRA: BlindRotationAlgo, BE: Backend> {
    fn bdd_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos;

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
    Self: CircuitBootstrappingKeyEncryptSk<BRA, BE> + GLWEToLWESwitchingKeyEncryptSk<BE> + GLWESwitchingKeyEncryptSk<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn bdd_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos,
    {
        self.circuit_bootstrapping_key_encrypt_sk_tmp_bytes(&infos.cbt_infos())
            .max(self.glwe_to_lwe_key_encrypt_sk_tmp_bytes(&infos.ks_lwe_infos()))
    }

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
        if let Some(key) = &mut res.ks_glwe {
            let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(sk_glwe.n(), key.rank_out());
            sk_out.fill_ternary_prob(0.5, source_xe);
            key.encrypt_sk(self, sk_glwe, &sk_out, source_xa, source_xe, scratch);
            res.ks_lwe
                .encrypt_sk(self, sk_lwe, &sk_out, source_xa, source_xe, scratch);
        } else {
            res.ks_lwe
                .encrypt_sk(self, sk_lwe, sk_glwe, source_xa, source_xe, scratch);
        }

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
    pub(crate) ks_glwe: Option<GLWESwitchingKeyPrepared<D, BE>>,
    pub(crate) ks_lwe: GLWEToLWEKeyPrepared<D, BE>,
}

impl<D: DataRef, BRA: BlindRotationAlgo, BE: Backend> BDDKeyInfos for BDDKeyPrepared<D, BRA, BE> {
    fn cbt_infos(&self) -> CircuitBootstrappingKeyLayout {
        CircuitBootstrappingKeyLayout {
            layout_brk: self.cbt.brk_infos(),
            layout_atk: self.cbt.atk_infos(),
            layout_tsk: self.cbt.tsk_infos(),
        }
    }
    fn ks_glwe_infos(&self) -> Option<GLWESwitchingKeyLayout> {
        self.ks_glwe.as_ref().map(|ks_glwe| GLWESwitchingKeyLayout {
            n: ks_glwe.n(),
            base2k: ks_glwe.base2k(),
            k: ks_glwe.k(),
            rank_in: ks_glwe.rank_in(),
            rank_out: ks_glwe.rank_out(),
            dnum: ks_glwe.dnum(),
            dsize: ks_glwe.dsize(),
        })
    }
    fn ks_lwe_infos(&self) -> GLWEToLWEKeyLayout {
        GLWEToLWEKeyLayout {
            n: self.ks_lwe.n(),
            base2k: self.ks_lwe.base2k(),
            k: self.ks_lwe.k(),
            rank_in: self.ks_lwe.rank_in(),
            dnum: self.ks_lwe.dnum(),
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
        let ks_glwe = if let Some(ks_glwe_infos) = &infos.ks_glwe_infos() {
            Some(GLWESwitchingKeyPrepared::alloc_from_infos(
                self,
                ks_glwe_infos,
            ))
        } else {
            None
        };

        BDDKeyPrepared {
            cbt: CircuitBootstrappingKeyPrepared::alloc_from_infos(self, &infos.cbt_infos()),
            ks_glwe,
            ks_lwe: GLWEToLWEKeyPrepared::alloc_from_infos(self, &infos.ks_lwe_infos()),
        }
    }

    fn prepare_bdd_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos,
    {
        self.circuit_bootstrapping_key_prepare_tmp_bytes(&infos.cbt_infos())
            .max(self.prepare_glwe_to_lwe_key_tmp_bytes(&infos.ks_lwe_infos()))
    }

    fn prepare_bdd_key<DM, DR>(&self, res: &mut BDDKeyPrepared<DM, BRA, BE>, other: &BDDKey<DR, BRA>, scratch: &mut Scratch<BE>)
    where
        DM: DataMut,
        DR: DataRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        res.cbt.prepare(self, &other.cbt, scratch);

        if let Some(key_prep) = &mut res.ks_glwe {
            if let Some(other) = &other.ks_glwe {
                key_prep.prepare(self, other, scratch);
            } else {
                panic!("incompatible keys: res has Some(ks_glwe) but other has none")
            }
        }

        res.ks_lwe.prepare(self, &other.ks_lwe, scratch);
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
        Option<&GLWESwitchingKeyPrepared<D, BE>>,
        &GLWEToLWEKeyPrepared<D, BE>,
    ) {
        (&self.cbt, self.ks_glwe.as_ref(), &self.ks_lwe)
    }
}

pub trait BDDKeyHelper<D: DataRef, BRA: BlindRotationAlgo, BE: Backend> {
    #[allow(clippy::type_complexity)]
    fn get_cbt_key(
        &self,
    ) -> (
        &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        Option<&GLWESwitchingKeyPrepared<D, BE>>,
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
