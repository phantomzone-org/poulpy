use std::collections::HashMap;

use poulpy_core::layouts::{GLWEAutomorphismKeyCompressed, GLWETensorKeyCompressed};
use poulpy_hal::layouts::Data;

use crate::tfhe::blind_rotation::{BlindRotationAlgo, BlindRotationKeyCompressed};

#[allow(dead_code)]
pub struct CircuitBootstrappingKey<D: Data, BRA: BlindRotationAlgo> {
    pub(crate) brk: BlindRotationKeyCompressed<D, BRA>,
    pub(crate) tsk: GLWETensorKeyCompressed<Vec<u8>>,
    pub(crate) atk: HashMap<i64, GLWEAutomorphismKeyCompressed<Vec<u8>>>,
}
