//! Secret-key and public-key encryption of ciphertexts and evaluation keys.
//!
//! This module provides traits and implementations for encrypting various
//! lattice-based cryptographic objects, including:
//!
//! - **Ciphertexts**: [`GLWEEncryptSk`], [`GLWEEncryptPk`], [`GGLWEEncryptSk`],
//!   [`GGSWEncryptSk`], [`LWEEncryptSk`] for encrypting plaintexts under
//!   GLWE, GGLWE, GGSW, and LWE schemes.
//!
//! - **Key-switching keys**: [`GLWESwitchingKeyEncryptSk`], [`LWESwitchingKeyEncrypt`],
//!   [`GLWEToLWESwitchingKeyEncryptSk`], [`LWEToGLWESwitchingKeyEncryptSk`] for
//!   generating keys that enable switching between different secret keys or
//!   between LWE and GLWE domains.
//!
//! - **Evaluation keys**: [`GLWEAutomorphismKeyEncryptSk`], [`GLWETensorKeyEncryptSk`],
//!   [`GGLWEToGGSWKeyEncryptSk`] for generating keys used in automorphism,
//!   tensor product, and GGLWE-to-GGSW conversion operations.
//!
//! - **Public keys**: [`GLWEPublicKeyGenerate`] for generating GLWE public keys
//!   from secret keys.
//!
//! Encryption methods follow a consistent pattern with PRNG sources:
//! - `source_xa`: source for mask/randomness sampling
//! - `source_xe`: source for error/noise sampling
//! - `source_xu`: source for uniform sampling (used in public-key encryption)
//!
//! Scratch space requirements for each operation can be queried via companion
//! `*_tmp_bytes` methods.

mod compressed;
mod gglwe;
mod gglwe_to_ggsw_key;
mod ggsw;
mod glwe;
mod glwe_automorphism_key;
mod glwe_public_key;
mod glwe_switching_key;
mod glwe_tensor_key;
mod glwe_to_lwe_key;
mod lwe;
mod lwe_switching_key;
mod lwe_to_glwe_key;

pub use compressed::*;
pub use gglwe::*;
pub use gglwe_to_ggsw_key::*;
pub use ggsw::*;
pub use glwe::*;
pub use glwe_automorphism_key::*;
pub use glwe_public_key::*;
pub use glwe_switching_key::*;
pub use glwe_tensor_key::*;
pub use glwe_to_lwe_key::*;
pub use lwe::*;
pub use lwe_switching_key::*;
pub use lwe_to_glwe_key::*;

/// Standard deviation of the discrete Gaussian distribution used for error sampling
/// during encryption. Set to 3.2.
pub const SIGMA: f64 = 3.2;

/// Truncation bound for the discrete Gaussian error distribution, defined as `6.0 * SIGMA`.
/// Samples are rejected if their absolute value exceeds this bound.
pub(crate) const SIGMA_BOUND: f64 = 6.0 * SIGMA;
