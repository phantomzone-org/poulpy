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
/// The compact plaintext is first placed into the ciphertext `offset_bits` position
/// of a full-width torus buffer. The GLWE encryption is then performed on the
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
    ct.assert_valid("encrypt_sk input ciphertext");
    ct.torus_scale_bits = pt.embed_bits;
    ct.offset_bits = ct.prefix_bits();
    ct.assert_valid("encrypt_sk metadata");
    let full_k = poulpy_core::layouts::TorusPrecision(ct.inner.base2k().0 * ct.inner.size() as u32);

    let layout = GLWEPlaintextLayout {
        n: ct.inner.n(),
        base2k: ct.inner.base2k(),
        k: full_k,
    };
    let (mut full_pt, scratch_rest) = scratch.take_glwe_plaintext(&layout);
    fill_offset_pt(
        module,
        &mut full_pt,
        poulpy_core::layouts::TorusPrecision(ct.offset_bits()),
        pt,
        scratch_rest,
    );

    // Encrypt on the full physical width so poulpy-core injects the fresh
    // error on the full torus buffer while the plaintext stays placed in the
    // semantic `offset_bits` position.
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
    let physical_k = ct.inner.base2k().0 * ct.inner.size() as u32;
    let pt_k = poulpy_core::layouts::TorusPrecision(physical_k.max(ct.offset_bits()));
    let full_pt_bytes = GLWEPlaintext::bytes_of(ct.inner.n(), ct.inner.base2k(), pt_k);
    full_pt_bytes
        + module
            .vec_znx_normalize_tmp_bytes()
            .max(GLWE::decrypt_tmp_bytes(module, &ct.inner))
}

/// Decrypts a [`CKKSCiphertext`] into a compact [`CKKSPlaintext`].
///
/// Decryption is performed on a width large enough to cover both the physical
/// limb prefix and the semantic message position `offset_bits`, then the compact
/// representation is extracted using the inverse placement for `offset_bits`.
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
    ct.assert_valid("decrypt input ciphertext");
    let ct_base2k = ct.inner.base2k();
    let physical_k = ct_base2k.0 * ct.inner.size() as u32;
    let pt_k = poulpy_core::layouts::TorusPrecision(physical_k.max(ct.offset_bits()));

    let layout = GLWEPlaintextLayout {
        n: ct.inner.n(),
        base2k: ct_base2k,
        k: pt_k,
    };
    let (mut full_pt, scratch_rest) = scratch.take_glwe_plaintext(&layout);
    ct.inner.decrypt(module, &mut full_pt, sk, scratch_rest);
    full_pt.set_k(pt_k);

    pt.embed_bits = ct.torus_scale_bits();
    extract_compact_pt(
        module,
        pt,
        poulpy_core::layouts::TorusPrecision(ct.offset_bits()),
        &full_pt,
        scratch_rest,
    );
}
