//! # poulpy-schemes
//!
//! Backend-agnostic implementations of mainstream RLWE-based Fully Homomorphic
//! Encryption (FHE) schemes, built on top of the low-level primitives provided
//! by `poulpy-core`, `poulpy-hal`, and the available compute backends
//! (`poulpy-cpu-ref`, `poulpy-cpu-avx`).
//!
//! ## Overview
//!
//! This crate provides three layered building blocks for constructing
//! gate-level and word-level FHE computation over encrypted binary data:
//!
//! - **Blind rotation** ([`bin_fhe::blind_rotation`]): Evaluation of a
//!   programmable lookup table (LUT) under a GLWE ciphertext, driven by an LWE
//!   ciphertext acting as a rotation index.  This is the fundamental gate
//!   bootstrapping primitive used throughout the rest of the crate.
//!
//! - **Circuit bootstrapping** ([`bin_fhe::circuit_bootstrapping`]): Conversion
//!   of a GLWE ciphertext encrypting a small plaintext into a GGSW ciphertext,
//!   enabling subsequent homomorphic selection (CMux) operations on ciphertexts.
//!   Circuit bootstrapping composes blind rotation, a trace/packing step, and a
//!   GGLWE-to-GGSW key-switch.
//!
//! - **BDD arithmetic** ([`bin_fhe::bdd_arithmetic`]): Word-level FHE arithmetic
//!   on encrypted unsigned integers ([`bin_fhe::bdd_arithmetic::FheUint`]) via
//!   statically compiled Binary Decision Diagram (BDD) circuits.  Operations
//!   such as addition, subtraction, bitwise logic, and shifts over `u32` are
//!   provided out of the box, evaluated bit-by-bit through GGSW-based CMux
//!   gates.
//!
//! ## Mental Model
//!
//! The computation model is:
//!
//! 1. A plaintext value (or each bit of a word) is encrypted as an LWE or GLWE
//!    ciphertext under a secret key.
//! 2. The ciphertext is *bootstrapped* — noise is refreshed and the value is
//!    simultaneously operated upon — by evaluating a programmable function
//!    encoded in a lookup table via blind rotation.
//! 3. For word-level arithmetic the bootstrapped bit values are converted to
//!    GGSW form (circuit bootstrapping) and then fed into a BDD circuit
//!    consisting of CMux gates.  The circuit outputs fresh GLWE ciphertexts
//!    encoding the result bits, which are then packed back into a single word
//!    ciphertext.
//!
//! ## Scheme Variants
//!
//! Only `bin_fhe` (binary/gate-level FHE) is currently exposed.  The algorithm
//! markers (e.g., [`bin_fhe::blind_rotation::CGGI`]) make the scheme identity
//! explicit in the type system, keeping key material and execution paths
//! unambiguous and allowing future variants to be added without breaking changes.
//!
//! ## Key Lifecycle
//!
//! Every key type exists in three forms:
//!
//! - **Standard** (`*Key`): Freshly encrypted raw key material.  Serialisable
//!   and suitable for storage or transport.
//! - **Compressed** (`*KeyCompressed`): A seed-based compact representation
//!   that regenerates the mask component on demand, halving storage.
//! - **Prepared** (`*KeyPrepared`): DFT-domain pre-processed form ready for
//!   fast on-line evaluation.  Not serialisable; must be derived from the
//!   standard form before evaluation.
//!
//! ## Scratch-Space Allocation
//!
//! All evaluation routines accept a mutable `Scratch<BE>` arena.  No heap
//! allocation occurs on the hot path.  Callers must allocate scratch space of
//! at least the size reported by the corresponding `*_tmp_bytes` method before
//! calling any evaluation function.
//!
//! ## Threading
//!
//! Several operations (BDD circuit evaluation, `FheUintPrepared` construction)
//! expose a `_multi_thread` variant that partitions independent output bits
//! across OS threads using `std::thread::scope`.  Each thread receives its own
//! slice of the scratch arena.  The module-level structures themselves are
//! `Sync`; the `Scratch` arena is not shared between threads.
//!
//! ## Non-Goals
//!
//! - This crate does not provide a high-level user-facing FHE API (no
//!   parameter generation, no automatic noise budget management).
//! - Ternary or higher-order plaintext spaces are not currently supported.
//! - Cryptographic parameter selection and security analysis are out of scope.
//!
//! ## Compatibility
//!
//! Requires a Rust edition 2024 toolchain.  Serialisation uses little-endian
//! binary encoding via the `byteorder` crate; format stability is not
//! guaranteed across minor versions.

pub mod bin_fhe;
