#[cfg(test)]
use poulpy_core::layouts::{
    AutomorphismKeyLayout, Base2K, Degree, Dnum, Dsize, GGSWLayout, GLWELayout, GLWEToLWEKeyLayout, Rank,
    TensorKeyLayout, TorusPrecision,
};

#[cfg(test)]
use crate::tfhe::{
    bdd_arithmetic::BDDKeyLayout, blind_rotation::BlindRotationKeyLayout, circuit_bootstrapping::CircuitBootstrappingKeyLayout,
};

#[cfg(test)]
pub(crate) const TEST_N_GLWE: u32 = 512;
#[cfg(test)]
pub(crate) const TEST_N_LWE: u32 = 77;
#[cfg(test)]
pub(crate) const TEST_BASE2K: u32 = 13;
#[cfg(test)]
pub(crate) const TEST_K_GLWE: u32 = 26;
#[cfg(test)]
pub(crate) const TEST_K_GGSW: u32 = 39;
#[cfg(test)]
pub(crate) const TEST_BLOCK_SIZE: u32 = 7;
#[cfg(test)]
pub(crate) const TEST_RANK: u32 = 2;

#[cfg(test)]
pub(crate) static TEST_GLWE_INFOS: GLWELayout = GLWELayout {
    n: Degree(TEST_N_GLWE),
    base2k: Base2K(TEST_BASE2K),
    k: TorusPrecision(TEST_K_GLWE),
    rank: Rank(TEST_RANK),
};

#[cfg(test)]
pub(crate) static TEST_GGSW_INFOS: GGSWLayout = GGSWLayout {
    n: Degree(TEST_N_GLWE),
    base2k: Base2K(TEST_BASE2K),
    k: TorusPrecision(TEST_K_GGSW),
    rank: Rank(TEST_RANK),
    dnum: Dnum(2),
    dsize: Dsize(1),
};

#[cfg(test)]
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
        layout_atk: AutomorphismKeyLayout {
            n: Degree(TEST_N_GLWE),
            base2k: Base2K(TEST_BASE2K),
            k: TorusPrecision(52),
            rank: Rank(TEST_RANK),
            dnum: Dnum(3),
            dsize: Dsize(1),
        },
        layout_tsk: TensorKeyLayout {
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
