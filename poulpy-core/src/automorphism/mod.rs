//! Galois automorphisms on lattice-based ciphertexts.
//!
//! This module provides traits and implementations for applying Galois automorphisms
//! to GLWE, GGSW, and GGLWE ciphertexts. A Galois automorphism is a ring morphism
//! on the cyclotomic polynomial ring Z\[X\]/(X^N + 1) defined by the mapping X -> X^k
//! for an odd integer k (the Galois element). These automorphisms permute the plaintext
//! slots of a ciphertext and are a key building block for operations such as slot rotations
//! and conjugation in FHE schemes.
//!
//! The automorphism is performed homomorphically using a GGLWE-based key (the automorphism
//! key), which internally combines a key-switching step with the polynomial automorphism.
//!
//! - [`GLWEAutomorphism`]: Automorphism on GLWE ciphertexts.
//! - [`GGSWAutomorphism`]: Automorphism on GGSW ciphertexts (applies GLWE automorphism
//!   row-wise then re-expands the GGSW structure).
//! - [`GLWEAutomorphismKeyAutomorphism`]: Automorphism on GGLWE automorphism keys themselves,
//!   composing two Galois elements.

mod gglwe_atk;
mod ggsw_ct;
mod glwe_ct;

pub use gglwe_atk::*;
pub use ggsw_ct::*;
pub use glwe_ct::*;
