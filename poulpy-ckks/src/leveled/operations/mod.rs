//! CKKS leveled operations.
//!
//! Each operation preserves the bivariate torus representation and updates the
//! ciphertext metadata (`k`, `log_delta`, `size`) consistently.
//!
//! ## Operations
//!
//! | Module | Operation |
//! |--------|-----------|
//! | [`add`] | Ciphertext addition (ct+ct, ct+pt, ct+const, prepared-pt variants) |
//! | [`sub`] | Ciphertext subtraction (same variants as add) |
//! | [`neg`] | Ciphertext negation |
//! | [`mul`] | Ciphertext multiplication (ct×ct, ct×pt, ct×const, prepared-pt, integer) |
//! | [`level`] | Rescale and metadata-preserving division by powers of two |
//! | [`rotate`] | Slot rotation via automorphism |
//! | [`conjugate`] | Complex conjugation via automorphism (Galois element `-1`) |

pub mod add;
pub mod conjugate;
pub mod level;
pub mod mul;
pub mod neg;
pub mod rotate;
pub mod sub;
pub(crate) mod utils;
