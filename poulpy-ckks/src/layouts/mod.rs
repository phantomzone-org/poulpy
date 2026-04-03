//! CKKS-level data structures.
//!
//! Each layout wraps the corresponding `poulpy-core` GLWE primitive and adds
//! the CKKS-specific metadata (`log_delta`) needed for leveled arithmetic.
//!
//! ## Key Structures
//!
//! | Type | Role |
//! |------|------|
//! | [`ciphertext::CKKSCiphertext`] | Encrypted CKKS value: GLWE ciphertext + `log_delta` |
//! | [`plaintext::CKKSPlaintext`] | Compact plaintext: integer polynomial with minimal limbs |
//! | [`plaintext_prepared::CKKSPlaintextPrepared`] | Pre-expanded plaintext for operations |
//! | [`tensor::CKKSTensor`] | Intermediate tensor product result between tensor and relinearize |
//! | [`keys`] | Relinearization and automorphism keys (standard + prepared) |

pub mod ciphertext;
pub mod keys;
pub mod plaintext;
pub mod plaintext_prepared;
pub mod tensor;
