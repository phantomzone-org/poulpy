//! CKKS ciphertext complex conjugation.
//!
//! Complex conjugation is the automorphism for Galois element `-1`.
//! The caller must supply the prepared automorphism key for that element.

use crate::layouts::ciphertext::CKKSCiphertext;
use poulpy_core::{GLWEAutomorphism, ScratchTakeCore, layouts::GLWEAutomorphismKeyPrepared};
use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

/// Returns the scratch bytes needed for [`conjugate`] and [`conjugate_inplace`].
pub fn conjugate_tmp_bytes<BE: Backend>(
    module: &Module<BE>,
    ct: &CKKSCiphertext<impl Data>,
    key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
) -> usize
where
    Module<BE>: GLWEAutomorphism<BE>,
{
    module.glwe_automorphism_tmp_bytes(&ct.inner, &ct.inner, key)
}

/// `res = Conjugate(ct)` using the conjugation key (Galois element -1).
pub fn conjugate<BE: Backend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<impl DataMut>,
    ct: &CKKSCiphertext<impl DataRef>,
    key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEAutomorphism<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    module.glwe_automorphism(&mut res.inner, &ct.inner, key, scratch);
    res.log_delta = ct.log_delta;
}

/// `ct = Conjugate(ct)` using the conjugation key (Galois element -1).
pub fn conjugate_inplace<BE: Backend>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertext<impl DataMut>,
    key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
    scratch: &mut Scratch<BE>,
) where
    Module<BE>: GLWEAutomorphism<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    module.glwe_automorphism_inplace(&mut ct.inner, key, scratch);
}
