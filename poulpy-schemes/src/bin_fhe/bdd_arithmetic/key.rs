use crate::bin_fhe::bdd_arithmetic::FheUintPreparedDebug;
use crate::bin_fhe::circuit_bootstrapping::CircuitBootstrappingKeyInfos;
use crate::bin_fhe::{
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

/// Dimension descriptor for a complete BDD evaluation key bundle.
///
/// Provides the layout parameters for the three constituent keys:
/// the circuit-bootstrapping key (`cbt`), the GLWE-to-LWE key-switching key
/// (`ks_lwe`), and the optional GLWE-to-GLWE key-switching key (`ks_glwe`).
///
/// `ks_glwe` is `Some` when the input ciphertext's GLWE rank differs from the
/// GLWE rank expected by the circuit-bootstrapping procedure, requiring an
/// intermediate rank reduction.
pub trait BDDKeyInfos {
    /// Layout of the circuit-bootstrapping key.
    fn cbt_infos(&self) -> CircuitBootstrappingKeyLayout;
    /// Layout of the GLWE-to-LWE key-switching key.
    fn ks_lwe_infos(&self) -> GLWEToLWEKeyLayout;
    /// Layout of the optional GLWE-to-GLWE key-switching key, or `None` if
    /// no intermediate rank reduction is needed.
    fn ks_glwe_infos(&self) -> Option<GLWESwitchingKeyLayout>;
}

/// Concrete dimension descriptor for a BDD evaluation key bundle.
///
/// Implements [`BDDKeyInfos`] and is suitable for use wherever a layout
/// descriptor is required (e.g. allocation, scratch-size queries).
#[derive(Debug, Clone, Copy)]
pub struct BDDKeyLayout {
    /// Layout of the circuit-bootstrapping key.
    pub cbt_layout: CircuitBootstrappingKeyLayout,
    /// Layout of the optional GLWE-to-GLWE key-switching key.
    pub ks_glwe_layout: Option<GLWESwitchingKeyLayout>,
    /// Layout of the GLWE-to-LWE key-switching key.
    pub ks_lwe_layout: GLWEToLWEKeyLayout,
}

impl BDDKeyInfos for BDDKeyLayout {
    fn cbt_infos(&self) -> CircuitBootstrappingKeyLayout {
        self.cbt_layout
    }

    fn ks_glwe_infos(&self) -> Option<GLWESwitchingKeyLayout> {
        self.ks_glwe_layout
    }

    fn ks_lwe_infos(&self) -> GLWEToLWEKeyLayout {
        self.ks_lwe_layout
    }
}

/// Raw BDD evaluation key bundle.
///
/// Contains the three sub-keys required to evaluate BDD circuits on encrypted
/// [`FheUint`] values:
///
/// - `cbt`: circuit-bootstrapping key (blind rotation + trace + key-switch).
/// - `ks_glwe`: optional GLWE-to-GLWE key-switching key for rank reduction before
///   LWE extraction.  Present when the input ciphertext's GLWE rank differs from
///   the bootstrapping GLWE rank.
/// - `ks_lwe`: GLWE-to-LWE key-switching key; applied after optional rank
///   reduction to produce LWE ciphertexts suitable for circuit bootstrapping.
///
/// ## Lifecycle
///
/// 1. Allocate with [`BDDKey::alloc_from_infos`].
/// 2. Fill with [`BDDKey::encrypt_sk`].
/// 3. Prepare into a [`BDDKeyPrepared`] before evaluation.
///
/// ## Thread Safety
///
/// `BDDKey` is `Sync`; multiple evaluation threads may hold shared references.
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
            ks_glwe: infos.ks_glwe_infos().as_ref().map(GLWESwitchingKey::alloc_from_infos),
            ks_lwe: GLWEToLWEKey::alloc_from_infos(&infos.ks_lwe_infos()),
        }
    }
}

/// Backend-level factory for encrypting a [`BDDKey`] under a secret key.
///
/// Implemented for `Module<BE>` when the backend supports circuit-bootstrapping
/// and switching-key encryption.  Callers should prefer the convenience method
/// [`BDDKey::encrypt_sk`].
pub trait BDDKeyEncryptSk<BRA: BlindRotationAlgo, BE: Backend> {
    /// Returns the minimum scratch-space size in bytes required by
    /// [`bdd_key_encrypt_sk`][Self::bdd_key_encrypt_sk].
    fn bdd_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos;

    /// Fills `res` with key material encrypted under `sk_lwe` / `sk_glwe`.
    ///
    /// `source_xa` supplies mask randomness; `source_xe` supplies error
    /// randomness.  The scratch arena must be at least
    /// [`bdd_key_encrypt_sk_tmp_bytes`][Self::bdd_key_encrypt_sk_tmp_bytes]
    /// bytes.
    ///
    /// When `res.ks_glwe` is `Some`, a fresh intermediate GLWE key is sampled
    /// from `source_xe` and used as the bridging secret; `ks_lwe` is then
    /// encrypted under that intermediate key rather than `sk_glwe` directly.
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
            res.ks_lwe.encrypt_sk(self, sk_lwe, &sk_out, source_xa, source_xe, scratch);
        } else {
            res.ks_lwe.encrypt_sk(self, sk_lwe, sk_glwe, source_xa, source_xe, scratch);
        }

        res.cbt.encrypt_sk(self, sk_lwe, sk_glwe, source_xa, source_xe, scratch);
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

/// DFT-prepared BDD evaluation key bundle, ready for on-line evaluation.
///
/// Mirrors the structure of [`BDDKey`] but stores all sub-keys in their
/// DFT (frequency) domain representations for fast matrix-vector products
/// during circuit bootstrapping and key-switching.
///
/// ## Invariants
///
/// - `ks_glwe` is `Some` if and only if the corresponding [`BDDKey`]'s
///   `ks_glwe` was `Some`.
///
/// ## Thread Safety
///
/// `BDDKeyPrepared<&[u8], BRA, BE>` is `Sync`; evaluation threads may share
/// a single prepared key while each holding their own scratch arena.
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
            brk_layout: self.cbt.brk_infos(),
            atk_layout: self.cbt.atk_infos(),
            tsk_layout: self.cbt.tsk_infos(),
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

/// Backend-level factory for allocating and preparing [`BDDKeyPrepared`] values.
///
/// Implemented for `Module<BE>` when the backend supports preparation of all
/// three constituent sub-keys.  Default method implementations delegate to
/// the corresponding sub-key factories.
pub trait BDDKeyPreparedFactory<BRA: BlindRotationAlgo, BE: Backend>
where
    Self: Sized + CircuitBootstrappingKeyPreparedFactory<BRA, BE> + GLWEToLWEKeyPreparedFactory<BE>,
{
    fn alloc_bdd_key_from_infos<A>(&self, infos: &A) -> BDDKeyPrepared<Vec<u8>, BRA, BE>
    where
        A: BDDKeyInfos,
    {
        let ks_glwe = if let Some(ks_glwe_infos) = &infos.ks_glwe_infos() {
            Some(GLWESwitchingKeyPrepared::alloc_from_infos(self, ks_glwe_infos))
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

/// Accessor trait for the constituent sub-keys of a prepared BDD key bundle.
///
/// Implemented by [`BDDKeyPrepared`].  Evaluation routines are generic over
/// this trait so that callers can pass any type that exposes the three
/// constituent prepared keys.
pub trait BDDKeyHelper<D: DataRef, BRA: BlindRotationAlgo, BE: Backend> {
    /// Returns references to the three constituent prepared keys in order:
    /// the circuit-bootstrapping key, the optional GLWE switching key, and
    /// the GLWE-to-LWE switching key.
    #[allow(clippy::type_complexity)]
    fn get_cbt_key(
        &self,
    ) -> (
        &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        Option<&GLWESwitchingKeyPrepared<D, BE>>,
        &GLWEToLWEKeyPrepared<D, BE>,
    );
}

/// Backend-level factory for building [`FheUintPreparedDebug`] values.
///
/// Unlike `FheUintPrepare`, this variant stores the per-bit GGSW ciphertexts
/// in standard (non-DFT) form, enabling noise inspection via
/// [`FheUintPreparedDebug::noise`] without a forward DFT transform.
pub trait FheUintPrepareDebug<BRA: BlindRotationAlgo, T: UnsignedInteger, BE: Backend> {
    /// Populates `res` by bootstrapping each bit of `bits` through `key`'s
    /// circuit-bootstrapping pipeline, storing the output GGSW in standard form.
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
