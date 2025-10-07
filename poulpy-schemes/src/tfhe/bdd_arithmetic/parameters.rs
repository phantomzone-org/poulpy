use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEAutomorphismKeyLayout, GGLWETensorKeyLayout, GLWEToLWEKeyLayout, Rank, TorusPrecision
};

use crate::tfhe::{
    bdd_arithmetic::BDDKeyLayout, blind_rotation::BlindRotationKeyLayout, circuit_bootstrapping::CircuitBootstrappingKeyLayout,
};

pub(crate) const TEST_N_GLWE: u32 = 1024;
pub(crate) const TEST_BASE2K: u32 = 13;
pub(crate) const TEST_K_GLWE: u32 = 26;
pub(crate) const TEST_K_GGSW: u32 = 39;
pub(crate) const TEST_BLOCK_SIZE: u32 = 7;

pub(crate) static TEST_BDD_KEY_LAYOUT: BDDKeyLayout = BDDKeyLayout {
    cbt: CircuitBootstrappingKeyLayout {
        layout_brk: BlindRotationKeyLayout {
            n_glwe: Degree(TEST_N_GLWE),
            n_lwe: Degree(574),
            base2k: Base2K(TEST_BASE2K),
            k: TorusPrecision(52),
            dnum: Dnum(3),
            rank: Rank(2),
        },
        layout_atk: GGLWEAutomorphismKeyLayout {
            n: Degree(TEST_N_GLWE),
            base2k: Base2K(TEST_BASE2K),
            k: TorusPrecision(52),
            rank: Rank(2),
            dnum: Dnum(3),
            dsize: Dsize(1),
        },
        layout_tsk: GGLWETensorKeyLayout {
            n: Degree(TEST_N_GLWE),
            base2k: Base2K(TEST_BASE2K),
            k: TorusPrecision(52),
            rank: Rank(2),
            dnum: Dnum(3),
            dsize: Dsize(1),
        },
    },
    ks: GLWEToLWEKeyLayout {
        n: Degree(TEST_N_GLWE),
        base2k: Base2K(TEST_BASE2K),
        k: TorusPrecision(39),
        dnum: Dnum(2),
        rank_in: Rank(2),
    },
};
