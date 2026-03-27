//! CKKS secret-key encryption and decryption.
//!
//! Thin wrappers over [`GLWE`] encryption and decryption that carry the
//! `log_delta` scale factor through the [`CKKSCiphertext`] /
//! [`CKKSPlaintext`] layout types.

use crate::layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext};
use poulpy_core::{
    GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{GLWE, prepared::GLWESecretPrepared},
};
use poulpy_hal::{
    api::ScratchTakeBasic,
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
    source::Source,
};

/// Returns the scratch-space size in bytes required by [`encrypt_sk`].
pub fn encrypt_sk_tmp_bytes<BE: Backend>(module: &Module<BE>, ct: &CKKSCiphertext<impl Data>) -> usize
where
    Module<BE>: GLWEEncryptSk<BE>,
{
    GLWE::encrypt_sk_tmp_bytes(module, &ct.inner)
}

/// Encrypts a CKKS plaintext under a GLWE secret key.
///
/// Produces a CKKS ciphertext `ct = (c0, c1)` such that decryption under
/// `sk` recovers `pt`. The `log_delta` scaling factor is propagated from
/// `pt` to `ct`.
///
/// - `pt`: the CKKS plaintext to encrypt.
/// - `sk`: the prepared GLWE secret key (DFT domain).
/// - `source_xa`: PRNG source for uniform mask sampling.
/// - `source_xe`: PRNG source for Gaussian error sampling.
/// - `scratch`: scratch space, sized by [`encrypt_sk_tmp_bytes`].
pub fn encrypt_sk<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    pt: &CKKSPlaintext<impl DataRef>,
    sk: &GLWESecretPrepared<impl DataRef, BE>,
    source_xa: &mut Source,
    source_xe: &mut Source,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEEncryptSk<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    ct.log_delta = pt.log_delta;
    ct.inner.encrypt_sk(module, &pt.inner, sk, source_xa, source_xe, scratch);
}

/// Returns the scratch-space size in bytes required by [`decrypt`].
pub fn decrypt_tmp_bytes<BE: Backend>(module: &Module<BE>, ct: &CKKSCiphertext<impl Data>) -> usize
where
    Module<BE>: GLWEDecrypt<BE>,
{
    GLWE::decrypt_tmp_bytes(module, &ct.inner)
}

/// Decrypts a CKKS ciphertext under a GLWE secret key.
///
/// Recovers the scaled polynomial from `ct` using `sk` and writes it to
/// `pt`. The `log_delta` scaling factor is propagated from `ct` to `pt`.
///
/// - `pt`: output CKKS plaintext.
/// - `ct`: the CKKS ciphertext to decrypt.
/// - `sk`: the prepared GLWE secret key (DFT domain).
/// - `scratch`: scratch space, sized by [`decrypt_tmp_bytes`].
pub fn decrypt<BE: Backend>(
    module: &Module<BE>,
    pt: &mut CKKSPlaintext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    sk: &GLWESecretPrepared<impl DataRef, BE>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEDecrypt<BE>,
    Scratch<BE>: ScratchTakeBasic,
{
    ct.inner.decrypt(module, &mut pt.inner, sk, scratch);
    pt.log_delta = ct.log_delta;
}
