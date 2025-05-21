pub mod automorphism;
pub mod elem;
pub mod gglwe_ciphertext;
pub mod ggsw_ciphertext;
pub mod glwe_ciphertext;
pub mod glwe_ciphertext_fourier;
pub mod glwe_plaintext;
pub mod keys;
pub mod keyswitch_key;
pub mod tensor_key;
#[cfg(test)]
mod test_fft64;
mod utils;

pub(crate) const SIX_SIGMA: f64 = 6.0;
