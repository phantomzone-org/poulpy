pub mod automorphism;
pub mod ciphertext;
pub mod decryption;
pub mod encryption;
pub mod external_product;
pub mod keyswitch;
pub mod ops;
pub mod packing;
pub mod plaintext;
pub mod public_key;
pub mod secret;
pub mod trace;

pub use ciphertext::GLWECiphertext;
pub(crate) use ciphertext::{GLWECiphertextToMut, GLWECiphertextToRef};
pub use ops::GLWEOps;
pub use packing::GLWEPacker;
pub use plaintext::GLWEPlaintext;
pub use public_key::GLWEPublicKey;
pub use secret::GLWESecret;

#[cfg(test)]
mod test_fft64;
