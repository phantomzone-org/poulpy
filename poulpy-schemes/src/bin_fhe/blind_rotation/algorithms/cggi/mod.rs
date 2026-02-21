mod algorithm;
mod key;
mod key_compressed;
mod key_prepared;

use crate::bin_fhe::blind_rotation::BlindRotationAlgo;

/// Algorithm marker for the
/// Chillotti-Gama-Georgieva-Izabachène (CGGI / TFHE) blind rotation.
///
/// `CGGI` is a zero-sized phantom type that selects the CGGI execution path
/// throughout the type system.  It implements [`BlindRotationAlgo`] and is
/// used as the `BRA` type parameter of `BlindRotationKey`,
/// `BlindRotationKeyPrepared`, and `BlindRotationExecute`.
///
/// Three concrete execution paths are dispatched at runtime based on the key
/// distribution stored in the prepared key:
///
/// - **`BinaryFixed` / `BinaryProb`** with `block_size == 1`: Classic CGGI —
///   one GGSW external product per LWE coefficient.
/// - **`BinaryBlock`** with `block_size > 1`: Block-CGGI — `block_size` LWE
///   coefficients are processed together with a shared DFT, reducing total
///   DFT evaluations.
/// - **`BinaryBlock`** with `extension_factor > 1`: Extended block-CGGI — the
///   lookup table spans multiple polynomials, increasing the effective domain
///   of the evaluated function.
#[derive(Clone)]
pub struct CGGI {}
impl BlindRotationAlgo for CGGI {}
