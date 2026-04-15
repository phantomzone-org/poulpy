//! # poulpy-ckks
//!
//! Backend-agnostic implementation of the CKKS (Cheon-Kim-Kim-Song)
//! homomorphic encryption scheme, built on top of the low-level primitives
//! provided by `poulpy-core`, `poulpy-hal`, and the available compute
//! backends (`poulpy-cpu-ref`, `poulpy-cpu-avx`).
//!
//! The crate uses a bivariate polynomial representation over the Torus
//! (base-`2^{base2k}` digits) instead of the RNS representation used by
//! most other CKKS libraries. A ciphertext tracks three related precisions:
//! the stored torus prefix `inner.k()`, the semantic message position
//! `offset_bits`, and the torus scaling factor `torus_scale_bits`. Rescale
//! visibly consumes precision by lowering all three together, while prefix
//! truncation lowers `inner.k()` and `offset_bits` without changing
//! `torus_scale_bits`.
//!
//! ## Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`layouts`] | CKKS-level data structures: ciphertext, plaintext, prepared plaintext, tensor, and evaluation keys |
//! | [`leveled`] | Leveled arithmetic (add, sub, mul, neg, rotate, conjugate), encryption, decryption, and rescale |
//! | [`bootstrapping`] | (Planned) CKKS bootstrapping |

use poulpy_core::layouts::{Base2K, TorusPrecision};

pub mod layouts;
pub mod leveled;
use anyhow::Result;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CKKS {
    /// Base 2 logarithm of the decimal precision.
    pub log_decimal: usize,
    /// Base 2 logarithm of the Remaining homomorphic capacity.
    pub log_hom_rem: usize,
}

impl CKKS {
    /// Returns the next multiple of [Base2K] greater than [Self::log_decimal] + [Self::log_hom_rem].
    pub fn min_k(&self, base2k: Base2K) -> TorusPrecision {
        ((self.log_decimal + self.log_hom_rem).next_multiple_of(base2k.as_usize())).into()
    }
}

pub trait CKKSInfos {
    fn meta(&self) -> CKKS;
    fn log_decimal(&self) -> usize;
    fn log_hom_rem(&self) -> usize;
    fn set_log_decimal(&mut self, log_decimal: usize) -> Result<()>;
    fn set_log_hom_rem(&mut self, log_hom_rem: usize) -> Result<()>;

    fn effective_k(&self) -> usize {
        self.log_decimal() + self.log_hom_rem()
    }
}
