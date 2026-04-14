//! CKKS-level data structures.
//!
//! Each layout wraps the corresponding `poulpy-core` GLWE primitive and adds
//! the CKKS-specific metadata needed for leveled arithmetic.
//!
//! ## Key Structures
//!
//! | Type | Role |
//! |------|------|
//! | `GLWE<D, CKKS>` | Encrypted CKKS value: GLWE ciphertext + offset / torus-scale metadata |

pub mod ciphertext;
pub mod plaintext;

pub use ciphertext::{CKKSMaintainOps, CKKSRescaleOps};
