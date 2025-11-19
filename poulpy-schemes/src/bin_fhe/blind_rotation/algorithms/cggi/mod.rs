mod algorithm;
mod key;
mod key_compressed;
mod key_prepared;

use crate::bin_fhe::blind_rotation::BlindRotationAlgo;

#[derive(Clone)]
pub struct CGGI {}
impl BlindRotationAlgo for CGGI {}
