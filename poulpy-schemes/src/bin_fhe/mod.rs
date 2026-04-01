//! Binary-gate FHE schemes.
//!
//! This module is the top-level entry point for the binary (gate-level) FHE
//! system implemented in this crate.  It organises the scheme into three
//! progressively higher-level layers:
//!
//! - [`blind_rotation`]: The foundational programmable-bootstrapping primitive.
//!   Given an LWE ciphertext as a rotation index and a pre-encoded lookup table,
//!   it produces a GLWE ciphertext whose plaintext coefficient is the function
//!   value at the encrypted index.
//!
//! - [`circuit_bootstrapping`]: Lifts a bootstrapped GLWE ciphertext into a
//!   GGSW ciphertext.  The result can be used as a selector for CMux gates,
//!   enabling arbitrary boolean circuits to be evaluated on encrypted data.
//!
//! - [`bdd_arithmetic`]: High-level word-level arithmetic on [`bdd_arithmetic::FheUint`]
//!   values.  Integer operations (add, sub, shift, compare, bitwise) are
//!   implemented as statically compiled BDD circuits evaluated through sequences
//!   of CMux gates, producing fresh packed-GLWE output ciphertexts.
//!
//! All sub-modules are parameterised by a `Backend` type (from `poulpy-hal`)
//! which selects the underlying arithmetic engine (reference CPU or AVX2).
pub mod bdd_arithmetic;
pub mod blind_rotation;
pub mod circuit_bootstrapping;
