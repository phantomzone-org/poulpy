//! CKKS evaluation key types.
//!
//! Each key type has a standard form (serializable, backend-agnostic) and a
//! prepared form (DFT-domain, backend-specific, used on the hot path).
//!
//! ## Key Structures
//!
//! | Type | Role |
//! |------|------|
//! | [`tensor_key::CKKSTensorKey`] | Relinearization key for ct × ct multiplication |
//! | [`tensor_key_prepared::CKKSTensorKeyPrepared`] | DFT-domain prepared relinearization key |
//! | [`automorphism_key::CKKSAutomorphismKey`] | Key for slot rotations and conjugation |
//! | [`automorphism_key_prepared::CKKSAutomorphismKeyPrepared`] | DFT-domain prepared automorphism key |

pub mod automorphism_key;
pub mod automorphism_key_prepared;
pub mod tensor_key;
pub mod tensor_key_prepared;
