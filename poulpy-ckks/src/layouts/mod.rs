//! CKKS-level data structures.
//!
//! Each layout wraps the corresponding `poulpy-core` GLWE primitive and adds
//! the CKKS-specific metadata needed for leveled arithmetic.
//!
//! ## Key Structures
//!
//! | Type | Role |
//! |------|------|
//! | `CKKSCiphertext<D>` | Encrypted CKKS value: CKKS wrapper over the core GLWE ciphertext |
//! | `CKKSPlaintextZnx<D>` | Quantized CKKS plaintext in the torus / ZNX domain |
//! | `CKKSPlaintextRnx<F>` | Floating-point CKKS plaintext in the RNX domain |

pub mod ciphertext;
pub mod plaintext;

pub use ciphertext::{CKKSCiphertext, CKKSMaintainOps};
pub use plaintext::{CKKSPlaintextConversion, CKKSPlaintextRnx, CKKSPlaintextZnx};
