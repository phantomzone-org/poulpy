mod add;
mod and;
mod fheuint;
mod ggsw_blind_rotations;
mod glwe_blind_rotation;
mod glwe_blind_selection;
mod or;
mod prepare;
mod sll;
mod slt;
mod sltu;
mod sra;
mod srl;
mod sub;
mod swap;
mod xor;

pub use add::*;
pub use and::*;
pub use fheuint::*;
pub use ggsw_blind_rotations::*;
pub use glwe_blind_rotation::*;
pub use glwe_blind_selection::*;
pub use or::*;
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};
pub use prepare::*;
pub use sll::*;
pub use slt::*;
pub use sltu::*;
pub use sra::*;
pub use srl::*;
pub use sub::*;
pub use swap::*;
pub use xor::*;

use poulpy_core::{
    ScratchTakeCore,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWEToGGSWKeyLayout, GGSWLayout, GLWEAutomorphismKeyLayout, GLWELayout, GLWESecret,
        GLWESecretPrepared, GLWESecretPreparedFactory, GLWESwitchingKeyLayout, GLWEToLWEKeyLayout, LWESecret, Rank,
        TorusPrecision,
    },
};

use crate::bin_fhe::{
    bdd_arithmetic::{BDDKey, BDDKeyEncryptSk, BDDKeyLayout, BDDKeyPrepared, BDDKeyPreparedFactory},
    blind_rotation::{
        BlindRotationAlgo, BlindRotationKey, BlindRotationKeyFactory, BlindRotationKeyLayout, BlindRotationKeyPreparedFactory,
    },
    circuit_bootstrapping::CircuitBootstrappingKeyLayout,
};

pub struct TestContext<BRA: BlindRotationAlgo, BE: Backend> {
    pub module: Module<BE>,
    pub sk_glwe: GLWESecretPrepared<Vec<u8>, BE>,
    pub sk_lwe: LWESecret<Vec<u8>>,
    pub bdd_key: BDDKeyPrepared<Vec<u8>, BRA, BE>,
}

impl<BRA: BlindRotationAlgo, BE: Backend> Default for TestContext<BRA, BE>
where
    Module<BE>: ModuleNew<BE>
        + BDDKeyEncryptSk<BRA, BE>
        + GLWESecretPreparedFactory<BE>
        + BlindRotationKeyPreparedFactory<BRA, BE>
        + BDDKeyPreparedFactory<BRA, BE>,
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend> TestContext<BRA, BE> {
    pub fn glwe_infos(&self) -> GLWELayout {
        TEST_GLWE_INFOS
    }

    pub fn ggsw_infos(&self) -> GGSWLayout {
        TEST_GGSW_INFOS
    }

    pub fn new() -> Self
    where
        Module<BE>: ModuleNew<BE>
            + BDDKeyEncryptSk<BRA, BE>
            + GLWESecretPreparedFactory<BE>
            + BlindRotationKeyPreparedFactory<BRA, BE>
            + BDDKeyPreparedFactory<BRA, BE>,
        BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let module: Module<BE> = Module::<BE>::new(TEST_N_GLWE as u64);

        let mut source_xs: Source = Source::new([1u8; 32]);
        let mut source_xa: Source = Source::new([2u8; 32]);
        let mut source_xe: Source = Source::new([3u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

        let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(TEST_N_GLWE.into(), TEST_RANK.into());
        sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_glwe_prep: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(&module, TEST_RANK.into());
        sk_glwe_prep.prepare(&module, &sk_glwe);

        let n_lwe: u32 = TEST_N_LWE;
        let block_size: u32 = TEST_BLOCK_SIZE;
        let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe.into());
        sk_lwe.fill_binary_block(block_size as usize, &mut source_xs);
        let bdd_key_infos: BDDKeyLayout = TEST_BDD_KEY_LAYOUT;
        let mut bdd_key: BDDKey<Vec<u8>, BRA> = BDDKey::alloc_from_infos(&bdd_key_infos);
        bdd_key.encrypt_sk(&module, &sk_lwe, &sk_glwe, &mut source_xa, &mut source_xe, scratch.borrow());
        let mut bdd_key_prepared: BDDKeyPrepared<Vec<u8>, BRA, BE> = BDDKeyPrepared::alloc_from_infos(&module, &bdd_key_infos);
        bdd_key_prepared.prepare(&module, &bdd_key, scratch.borrow());

        TestContext {
            bdd_key: bdd_key_prepared,
            sk_glwe: sk_glwe_prep,
            sk_lwe,
            module,
        }
    }
}

pub(crate) const TEST_N_GLWE: u32 = 256;
pub(crate) const TEST_N_LWE: u32 = 77;
pub(crate) const TEST_FHEUINT_BASE2K: u32 = 13;
pub(crate) const TEST_BRK_BASE2K: u32 = 12;
pub(crate) const TEST_ATK_BASE2K: u32 = 11;
pub(crate) const TEST_TSK_BASE2K: u32 = 10;
pub(crate) const TEST_LWE_BASE2K: u32 = 4;
pub(crate) const TEST_K_GLWE: u32 = 26;
pub(crate) const TEST_K_GGSW: u32 = 39;
pub(crate) const TEST_BLOCK_SIZE: u32 = 7;
pub(crate) const TEST_RANK: u32 = 2;

pub(crate) static TEST_GLWE_INFOS: GLWELayout = GLWELayout {
    n: Degree(TEST_N_GLWE),
    base2k: Base2K(TEST_FHEUINT_BASE2K),
    k: TorusPrecision(TEST_K_GLWE),
    rank: Rank(TEST_RANK),
};

pub(crate) static TEST_GGSW_INFOS: GGSWLayout = GGSWLayout {
    n: Degree(TEST_N_GLWE),
    base2k: Base2K(TEST_FHEUINT_BASE2K),
    k: TorusPrecision(TEST_K_GGSW),
    rank: Rank(TEST_RANK),
    dnum: Dnum(2),
    dsize: Dsize(1),
};

pub(crate) static TEST_BDD_KEY_LAYOUT: BDDKeyLayout = BDDKeyLayout {
    cbt_layout: CircuitBootstrappingKeyLayout {
        brk_layout: BlindRotationKeyLayout {
            n_glwe: Degree(TEST_N_GLWE),
            n_lwe: Degree(TEST_N_LWE),
            base2k: Base2K(TEST_BRK_BASE2K),
            k: TorusPrecision(52),
            dnum: Dnum(4),
            rank: Rank(TEST_RANK),
        },
        atk_layout: GLWEAutomorphismKeyLayout {
            n: Degree(TEST_N_GLWE),
            base2k: Base2K(TEST_ATK_BASE2K),
            k: TorusPrecision(52),
            rank: Rank(TEST_RANK),
            dnum: Dnum(4),
            dsize: Dsize(1),
        },
        tsk_layout: GGLWEToGGSWKeyLayout {
            n: Degree(TEST_N_GLWE),
            base2k: Base2K(TEST_TSK_BASE2K),
            k: TorusPrecision(52),
            rank: Rank(TEST_RANK),
            dnum: Dnum(4),
            dsize: Dsize(1),
        },
    },
    ks_glwe_layout: Some(GLWESwitchingKeyLayout {
        n: Degree(TEST_N_GLWE),
        base2k: Base2K(TEST_LWE_BASE2K),
        k: TorusPrecision(20),
        rank_in: Rank(TEST_RANK),
        rank_out: Rank(1),
        dnum: Dnum(3),
        dsize: Dsize(1),
    }),
    ks_lwe_layout: GLWEToLWEKeyLayout {
        n: Degree(TEST_N_GLWE),
        base2k: Base2K(TEST_LWE_BASE2K),
        k: TorusPrecision(16),
        rank_in: Rank(1),
        dnum: Dnum(3),
    },
};
