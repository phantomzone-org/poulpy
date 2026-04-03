//! CKKS secret-key encryption and decryption.
//!
//! Encryption expands a compact [`CKKSPlaintext`] into the ciphertext's full
//! torus width, places the message inside the active `k`-bit window, then
//! calls the underlying GLWE encryption.  Decryption reverses the process:
//! decrypt into a full torus plaintext, then extract the compact representation
//! matching the ciphertext's active `k`.
//!
//! The core GLWE encryption is invoked with the physical width
//! so that noise is injected across the full torus buffer, while the message
//! is positioned according to the semantic precision `k`.

use crate::{
    layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext},
    leveled::operations::utils::{extract_compact_pt, fill_offset_pt},
};
use poulpy_core::{
    GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{GLWE, GLWEPlaintext, GLWEPlaintextLayout, LWEInfos, SetGLWEInfos, prepared::GLWESecretPrepared},
};
use poulpy_hal::{
    api::{VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
    source::Source,
};

/// Returns the scratch bytes needed for [`encrypt_sk`].
pub fn encrypt_sk_tmp_bytes<BE: Backend>(module: &Module<BE>, ct: &CKKSCiphertext<impl Data>) -> usize
where
    Module<BE>: GLWEEncryptSk<BE> + VecZnxNormalizeTmpBytes,
{
    let full_k = poulpy_core::layouts::TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32);
    let layout_bytes = GLWEPlaintext::bytes_of(ct.inner.n(), ct.inner.base2k(), full_k);
    layout_bytes
        + module
            .vec_znx_normalize_tmp_bytes()
            .max(GLWE::encrypt_sk_tmp_bytes(module, &ct.inner))
}

/// Encrypts a compact [`CKKSPlaintext`] under a GLWE secret key.
///
/// The compact plaintext is first placed into the active `k`-bit window of
/// a full-width torus buffer.  The GLWE encryption is then performed on the
/// full physical width so that fresh noise covers the entire representation.
pub fn encrypt_sk<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    pt: &CKKSPlaintext<impl DataRef>,
    sk: &GLWESecretPrepared<impl DataRef, BE>,
    source_xa: &mut Source,
    source_xe: &mut Source,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEEncryptSk<BE> + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    ct.log_delta = pt.log_delta;
    let full_k = poulpy_core::layouts::TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32);

    let layout = GLWEPlaintextLayout {
        n: ct.inner.n(),
        base2k: ct.inner.base2k(),
        k: full_k,
    };
    let (mut full_pt, scratch_rest) = scratch.take_glwe_plaintext(&layout);
    fill_offset_pt(module, &mut full_pt, ct.inner.k(), pt, scratch_rest);

    // Encrypt on the full physical width so poulpy-core injects the fresh
    // error on the full torus buffer while the plaintext stays placed in the
    // active `k` window.
    let actual_k = ct.inner.k();
    let max_k = full_k;
    ct.inner.set_k(max_k);
    ct.inner.encrypt_sk(module, &full_pt, sk, source_xa, source_xe, scratch_rest);
    ct.inner.set_k(actual_k);
}

/// Returns the scratch bytes needed for [`decrypt`].
pub fn decrypt_tmp_bytes<BE: Backend>(module: &Module<BE>, ct: &CKKSCiphertext<impl Data>) -> usize
where
    Module<BE>: GLWEDecrypt<BE> + VecZnxNormalizeTmpBytes,
{
    let full_k = poulpy_core::layouts::TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32);
    let full_pt_bytes = GLWEPlaintext::bytes_of(ct.inner.n(), ct.inner.base2k(), full_k);
    full_pt_bytes
        + module
            .vec_znx_normalize_tmp_bytes()
            .max(GLWE::decrypt_tmp_bytes(module, &ct.inner))
}

/// Decrypts a [`CKKSCiphertext`] into a compact [`CKKSPlaintext`].
///
/// Decryption is performed on the full physical width, then the compact
/// representation is extracted using the inverse placement for the
/// ciphertext's active `k`.
pub fn decrypt<BE: Backend>(
    module: &Module<BE>,
    pt: &mut CKKSPlaintext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    sk: &GLWESecretPrepared<impl DataRef, BE>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEDecrypt<BE> + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let ct_base2k = ct.inner.base2k();
    let pt_k = poulpy_core::layouts::TorusPrecision(ct_base2k.0 * ct.inner.size() as u32);

    let layout = GLWEPlaintextLayout {
        n: ct.inner.n(),
        base2k: ct_base2k,
        k: pt_k,
    };
    let (mut full_pt, scratch_rest) = scratch.take_glwe_plaintext(&layout);
    ct.inner.decrypt(module, &mut full_pt, sk, scratch_rest);

    pt.log_delta = ct.log_delta;
    extract_compact_pt(module, pt, ct.inner.k(), &full_pt, scratch_rest);
}
