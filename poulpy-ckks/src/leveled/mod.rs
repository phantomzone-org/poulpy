//! Leveled CKKS arithmetic, encryption, and decryption.
//!
//! This module provides the core leveled evaluation pipeline:
//!
//! - [`encryption`]: secret-key encrypt / decrypt with explicit torus placement.
//! - [`operations`]: add, sub, neg, mul, rotate, conjugate, and rescale.
//!
//! All hot-path operations use scratch-based allocation; no heap allocation
//! occurs during leveled arithmetic.

pub mod encryption;
pub mod operations;
pub mod tests;
