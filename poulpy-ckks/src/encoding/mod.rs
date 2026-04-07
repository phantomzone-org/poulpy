//! CKKS encoding and decoding.
//!
//! This module implements the encoding and decoding routines for the
//! (CKKS) homomorphic encryption scheme.
//!
//! ## Encoding Variants
//!
//! | Function prefix | Description |
//! |-----------------|-------------|
//! | [`classical::encode`] | Standard CKKS encoding via inverse canonical embedding. |
//! | `encode_ci` | (Planned) Encoding into the real subring $Z[X+X^{-1}]$. |

pub mod classical;
