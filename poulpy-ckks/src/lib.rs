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
//! | [`encoding`] | Classical CKKS encoding / decoding via the canonical embedding (inverse FFT) |
//! | [`leveled`] | Leveled arithmetic (add, sub, mul, neg, rotate, conjugate), encryption, decryption, and rescale |
//! | [`bootstrapping`] | (Planned) CKKS bootstrapping |

pub mod encoding;
pub mod layouts;
pub mod leveled;
