//! CKKS-level data structures.
//!
//! Each layout wraps the corresponding `poulpy-core` GLWE primitive and adds
//! the CKKS-specific metadata needed for leveled arithmetic.
//!
//! ## Key Structures
//!
//! | Type | Role |
//! |------|------|
//! | [`ciphertext::CKKSCiphertext`] | Encrypted CKKS value: GLWE ciphertext + offset / torus-scale metadata |

use anyhow::Result;
use poulpy_core::layouts::{Base2K, TorusPrecision};

pub mod ciphertext;
pub mod plaintext;

#[derive(Debug, Clone, Copy, Default)]
pub struct Metadata {
    /// Base 2 logarithm of the decimal precision.
    pub log_decimal: usize,
    /// Base 2 logarithm of the Remaining homomorphic capacity.
    pub log_hom_rem: usize,
}

impl Metadata {
    /// Returns the next multiple of [Base2K] greater than [Self::log_decimal] + [Self::log_hom_rem].
    pub fn min_k(&self, base2k: Base2K) -> TorusPrecision {
        ((self.log_decimal + self.log_hom_rem).next_multiple_of(base2k.as_usize())).into()
    }
}

pub trait PrecisionInfos {
    fn log_decimal(&self) -> usize;
    fn log_hom_rem(&self) -> usize;
    fn set_log_decimal(&mut self, log_decimal: usize) -> Result<()>;
    fn set_log_hom_rem(&mut self, log_integer: usize) -> Result<()>;

    fn effective_k(&self) -> TorusPrecision {
        (self.log_decimal() + self.log_hom_rem()).into()
    }
}
