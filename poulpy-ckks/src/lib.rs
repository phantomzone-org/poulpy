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
//! | [`encoding`] | CKKS encoders/decoders, including slot-wise real/imaginary packing |
//! | [`layouts`] | CKKS-level data structures: ciphertext, plaintext, prepared plaintext, tensor, and evaluation keys |
//! | [`leveled`] | Leveled arithmetic (add, sub, mul, neg, rotate, conjugate), encryption, decryption, and rescale |
//! | [`bootstrapping`] | (Planned) CKKS bootstrapping |

use poulpy_core::layouts::{Base2K, TorusPrecision};

pub mod encoding;
mod error;
pub mod layouts;
pub mod leveled;
pub use error::CKKSCompositionError;
pub(crate) use error::{checked_log_hom_rem_sub, checked_mul_ct_log_hom_rem, ensure_base2k_match, ensure_plaintext_alignment};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// CKKS semantic precision metadata carried by ciphertexts and plaintexts.
///
/// `log_decimal` is the scaling precision of the encoded value and
/// `log_hom_rem` is the remaining homomorphic headroom available before the
/// value must be rescaled or truncated.
pub struct CKKSMeta {
    /// Base 2 logarithm of the decimal precision.
    pub log_decimal: usize,
    /// Base 2 logarithm of the Remaining homomorphic capacity.
    pub log_hom_rem: usize,
}

/// Common metadata accessors for CKKS ciphertext and plaintext containers.
///
/// This trait exposes the semantic precision of a value independently from the
/// raw limb storage used by the underlying torus representation.
pub trait CKKSInfos {
    /// Returns the complete metadata pair.
    fn meta(&self) -> CKKSMeta;

    /// Returns the base-2 logarithm of the encoded decimal scaling factor.
    fn log_decimal(&self) -> usize;

    /// Returns the base-2 logarithm of the remaining homomorphic capacity.
    fn log_hom_rem(&self) -> usize;

    /// Returns the next multiple of [Base2K] greater than [Self::log_decimal] + [Self::log_hom_rem].
    fn min_k(&self, base2k: Base2K) -> TorusPrecision {
        ((self.log_decimal() + self.log_hom_rem()).next_multiple_of(base2k.as_usize())).into()
    }

    /// Returns the semantic torus width carried by the value.
    ///
    /// This is `log_decimal + log_hom_rem` and may differ from the rounded
    /// storage capacity `max_k()`.
    fn effective_k(&self) -> usize {
        self.log_decimal() + self.log_hom_rem()
    }
}

impl CKKSInfos for CKKSMeta {
    fn meta(&self) -> CKKSMeta {
        *self
    }

    fn log_decimal(&self) -> usize {
        self.log_decimal
    }

    fn log_hom_rem(&self) -> usize {
        self.log_hom_rem
    }
}
