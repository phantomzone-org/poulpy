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
//!
//! ## End-to-end Example
//!
//! For a runnable end-to-end example, see `examples/poly2.rs`.
//! It encrypts `x`, evaluates `a + bx + cx^2`, then decrypts and decodes the
//! resulting slots.
//!
//! The evaluator portion looks like:
//!
//! ```rust,ignore
//! let pt_x = encode_pt_znx::<NTT120Ref>(&encoder, &x_re, &x_im, prec)?;
//! let ct_x = encrypt(&module, &sk, &pt_x, scratch.borrow())?;
//!
//! let mut ct_x2 = CKKSCiphertext::alloc(n.into(), ct_x.log_hom_rem().into(), base2k.into());
//! module.ckks_square(&mut ct_x2, &ct_x, &tsk_prepared, scratch.borrow())?;
//!
//! let mut term_bx = CKKSCiphertext::alloc(n.into(), ct_x.log_hom_rem().into(), base2k.into());
//! module.ckks_mul_pt_vec_rnx(&mut term_bx, &ct_x, &pt_b, prec, scratch.borrow())?;
//!
//! let mut term_cx2 = CKKSCiphertext::alloc(n.into(), ct_x2.log_hom_rem().into(), base2k.into());
//! module.ckks_mul_pt_vec_rnx(&mut term_cx2, &ct_x2, &pt_c, prec, scratch.borrow())?;
//!
//! let mut poly = CKKSCiphertext::alloc(n.into(), term_cx2.effective_k().into(), base2k.into());
//! module.ckks_add(&mut poly, &term_bx, &term_cx2, scratch.borrow())?;
//! module.ckks_add_pt_vec_rnx_inplace(&mut poly, &pt_a, prec, scratch.borrow())?;
//! ```

use poulpy_core::layouts::{Base2K, TorusPrecision};

pub mod encoding;
mod error;
pub mod layouts;
pub mod leveled;
pub use error::CKKSCompositionError;
pub(crate) use error::{
    checked_log_hom_rem_sub, checked_mul_ct_log_hom_rem, ensure_base2k_match, ensure_limb_count_fits, ensure_plaintext_alignment,
};

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
    fn effective_k(&self) -> usize {
        self.log_decimal() + self.log_hom_rem()
    }
}

impl CKKSInfos for CKKS {
    fn meta(&self) -> CKKS {
        *self
    }

    fn log_decimal(&self) -> usize {
        self.log_decimal
    }

    fn log_hom_rem(&self) -> usize {
        self.log_hom_rem
    }
}
