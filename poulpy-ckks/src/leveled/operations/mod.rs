//! CKKS leveled operations.
//!
//! Each operation preserves the bivariate torus representation and updates the
//! ciphertext metadata (`k`, `offset_bits`, `torus_scale_bits`, `size`) consistently.
//!
//! ## Operations
//!
//! | Module | Operation |
//! |--------|-----------|
//! | [`add`] | Ciphertext addition (ct+ct, ct+pt in ZNX form, ct+pt in RNX form) |
//! | [`sub`] | Ciphertext subtraction (same variants as add) |
//! | [`neg`] | Ciphertext negation |
//! | [`mul`] | Ciphertext multiplication (ct×ct, ct×pt, ct×const, prepared-pt, integer) |
//! | [`pow2`] | Metadata-preserving multiplication and division by powers of two |
//! | [`rotate`] | Slot rotation via automorphism |
//! | [`conjugate`] | Complex conjugation via automorphism (Galois element `-1`) |

pub mod add;
pub mod conjugate;
pub mod mul;
pub mod neg;
pub mod pow2;
pub mod pt_znx;
pub mod rotate;
pub mod sub;
