//! CKKS secret-key encryption and decryption.
//!
//! Encryption expands a compact [`CKKSPlaintext`] into the ciphertext's full
//! torus width, places the message according to the ciphertext message position,
//! then calls the underlying GLWE encryption. Decryption reverses the process:
//! decrypt into a sufficiently wide torus plaintext, then extract the compact
//! representation using the ciphertext `offset_bits`.
//!
//! The core GLWE encryption is invoked with the physical width
//! so that noise is injected across the full torus buffer, while the message
//! is positioned according to the semantic precision `offset_bits`.

use crate::layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintextZnx};
use poulpy_core::{
    EncryptionInfos, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{GLWE, GLWEInfos, GLWEPlaintext, LWEInfos, prepared::GLWESecretPrepared},
};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxRshAdd},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

/// Returns the scratch bytes needed for [`encrypt_sk`].
pub fn encrypt_sk_tmp_bytes<A, BE: Backend>(module: &Module<BE>, ct_infos: &A) -> usize
where
    Module<BE>: GLWEEncryptSk<BE> + VecZnxNormalizeTmpBytes,
    A: GLWEInfos,
{
    GLWEPlaintext::bytes_of_from_infos(ct_infos)
        + module
            .vec_znx_normalize_tmp_bytes()
            .max(GLWE::encrypt_sk_tmp_bytes(module, ct_infos))
}

/// Encrypts a compact [`CKKSPlaintext`] under a GLWE secret key.
///
/// The compact plaintext is first placed into the ciphertext `offset_bits` position
/// of a full-width torus buffer. The GLWE encryption is then performed on the
/// full physical width so that fresh noise covers the entire representation.
pub fn encrypt_sk<BE: Backend, E: EncryptionInfos>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    pt: &CKKSPlaintextZnx<impl DataRef>,
    sk: &GLWESecretPrepared<impl DataRef, BE>,
    enc_infos: &E,
    source_xa: &mut Source,
    source_xe: &mut Source,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEEncryptSk<BE> + VecZnxNormalize<BE> + VecZnxRshAdd<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    ct.inner.encrypt_zero_sk(module, sk, enc_infos, source_xe, source_xa, scratch);
    let log_delta = enc_infos.noise_infos().k - ct.inner.base2k().as_usize();
    pt.add_to(module, ct.inner.data_mut(), log_delta, scratch);
    ct.log_delta = log_delta;
}

/// Returns the scratch bytes needed for [`decrypt`].
pub fn decrypt_tmp_bytes<A, BE: Backend>(module: &Module<BE>, ct_infos: &A) -> usize
where
    Module<BE>: GLWEDecrypt<BE>,
    A: GLWEInfos,
{
    GLWEPlaintext::bytes_of_from_infos(ct_infos) + module.glwe_decrypt_tmp_bytes(ct_infos)
}

/// Decrypts a [`CKKSCiphertext`] into a compact [`CKKSPlaintext`].
///
/// Decryption is performed on a width large enough to cover both the physical
/// limb prefix and the semantic message position `offset_bits`, then the compact
/// representation is extracted using the inverse placement for `offset_bits`.
pub fn decrypt<BE: Backend>(
    module: &Module<BE>,
    pt: &mut CKKSPlaintextZnx<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    sk: &GLWESecretPrepared<impl DataRef, BE>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEDecrypt<BE> + VecZnxNormalize<BE> + VecZnxLsh<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let (mut full_pt, scratch_rest) = scratch.take_glwe_plaintext(&ct.inner);
    ct.inner.decrypt(module, &mut full_pt, sk, scratch_rest);
    pt.extract_from(module, &full_pt.data, ct.log_delta, scratch_rest);
}
