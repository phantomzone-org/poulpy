mod add;
mod and;
mod ggsw_blind_rotations;
mod glwe_blind_rotation;
mod or;
mod prepare;
mod sll;
mod slt;
mod sltu;
mod sra;
mod srl;
mod sub;
mod xor;

pub use add::*;
pub use and::*;
pub use ggsw_blind_rotations::*;
pub use glwe_blind_rotation::*;
pub use or::*;
pub use prepare::*;
pub use sll::*;
pub use slt::*;
pub use sltu::*;
pub use sra::*;
pub use srl::*;
pub use sub::*;
pub use xor::*;

use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGSWLayout, GLWEAutomorphismKeyLayout, GLWELayout, GLWETensorKeyLayout, GLWEToLWEKeyLayout,
    Rank, TorusPrecision,
};

use crate::tfhe::{
    bdd_arithmetic::BDDKeyLayout, blind_rotation::BlindRotationKeyLayout, circuit_bootstrapping::CircuitBootstrappingKeyLayout,
};

pub(crate) const TEST_N_GLWE: u32 = 512;
pub(crate) const TEST_N_LWE: u32 = 77;
pub(crate) const TEST_BASE2K: u32 = 13;
pub(crate) const TEST_K_GLWE: u32 = 26;
pub(crate) const TEST_K_GGSW: u32 = 39;
pub(crate) const TEST_BLOCK_SIZE: u32 = 7;
pub(crate) const TEST_RANK: u32 = 2;

pub(crate) static TEST_GLWE_INFOS: GLWELayout = GLWELayout {
    n: Degree(TEST_N_GLWE),
    base2k: Base2K(TEST_BASE2K),
    k: TorusPrecision(TEST_K_GLWE),
    rank: Rank(TEST_RANK),
};

pub(crate) static TEST_GGSW_INFOS: GGSWLayout = GGSWLayout {
    n: Degree(TEST_N_GLWE),
    base2k: Base2K(TEST_BASE2K),
    k: TorusPrecision(TEST_K_GGSW),
    rank: Rank(TEST_RANK),
    dnum: Dnum(2),
    dsize: Dsize(1),
};

pub(crate) static TEST_BDD_KEY_LAYOUT: BDDKeyLayout = BDDKeyLayout {
    cbt: CircuitBootstrappingKeyLayout {
        layout_brk: BlindRotationKeyLayout {
            n_glwe: Degree(TEST_N_GLWE),
            n_lwe: Degree(TEST_N_LWE),
            base2k: Base2K(TEST_BASE2K),
            k: TorusPrecision(52),
            dnum: Dnum(3),
            rank: Rank(TEST_RANK),
        },
        layout_atk: GLWEAutomorphismKeyLayout {
            n: Degree(TEST_N_GLWE),
            base2k: Base2K(TEST_BASE2K),
            k: TorusPrecision(52),
            rank: Rank(TEST_RANK),
            dnum: Dnum(3),
            dsize: Dsize(1),
        },
        layout_tsk: GLWETensorKeyLayout {
            n: Degree(TEST_N_GLWE),
            base2k: Base2K(TEST_BASE2K),
            k: TorusPrecision(52),
            rank: Rank(TEST_RANK),
            dnum: Dnum(3),
            dsize: Dsize(1),
        },
    },
    ks: GLWEToLWEKeyLayout {
        n: Degree(TEST_N_GLWE),
        base2k: Base2K(TEST_BASE2K),
        k: TorusPrecision(39),
        rank_in: Rank(TEST_RANK),
        dnum: Dnum(2),
    },
};
