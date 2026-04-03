//! CKKS ciphertext slot rotation.
//!
//! Slot rotation is an automorphism whose Galois element is determined by
//! the rotation amount.  The caller must supply the prepared automorphism
//! key for the desired rotation (obtained via `Module::galois_element`).

use crate::layouts::ciphertext::CKKSCiphertext;
use poulpy_core::{GLWEAutomorphism, ScratchTakeCore, layouts::GLWEAutomorphismKeyPrepared};
use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

/// Returns the scratch bytes needed for [`rotate`] and [`rotate_inplace`].
pub fn rotate_tmp_bytes<BE: Backend>(
    module: &Module<BE>,
    ct: &CKKSCiphertext<impl Data>,
    key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
) -> usize
where
    Module<BE>: GLWEAutomorphism<BE>,
{
    module.glwe_automorphism_tmp_bytes(&ct.inner, &ct.inner, key)
}

/// `res = Rotate(ct)` by the rotation encoded in `key`.
pub fn rotate<BE: Backend>(
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

/// `ct = Rotate(ct)` by the rotation encoded in `key`.
pub fn rotate_inplace<BE: Backend>(
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
