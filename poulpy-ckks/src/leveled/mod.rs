//! Leveled CKKS arithmetic, encryption, and decryption.
//!
//! This module provides the core leveled evaluation pipeline:
//!
//! - [`encryption`]: secret-key encrypt / decrypt with explicit torus placement.
//! - [`operations`]: add, sub, neg, mul, rotate, conjugate, and rescale.
//! - [`rescale`]: explicit rescaling and level-alignment helpers.
//! - [`tmp_bytes`]: aggregate scratch-size helpers covering broad CKKS workflows.
//!
//! All hot-path operations use scratch-based allocation; no heap allocation
//! occurs during leveled arithmetic.

pub mod encryption;
pub mod operations;
pub mod rescale;
pub mod tests;
pub mod tmp_bytes;

pub use tmp_bytes::CKKSAllOpsTmpBytes;
