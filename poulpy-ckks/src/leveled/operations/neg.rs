//! CKKS ciphertext negation.
//!
//! Negates each column of the GLWE ciphertext. `torus_scale_bits` is preserved.

use crate::layouts::ciphertext::CKKSCiphertext;
use poulpy_hal::{
    api::{VecZnxNegate, VecZnxNegateInplace},
    layouts::{Backend, DataMut, DataRef, Module},
};

/// Computes `res = -ct`.
pub fn neg<BE: Backend>(module: &Module<BE>, res: &mut CKKSCiphertext<impl DataMut>, ct: &CKKSCiphertext<impl DataRef>)
where
    Module<BE>: VecZnxNegate,
{
    ct.assert_valid("neg input");
    res.torus_scale_bits = ct.torus_scale_bits();
    res.offset_bits = ct.offset_bits();
    let ncols = ct.inner.data().cols;
    for i in 0..ncols {
        module.vec_znx_negate(res.inner.data_mut(), i, ct.inner.data(), i);
    }
    res.assert_valid("neg result");
}

/// Computes `ct = -ct` in place.
pub fn neg_inplace<BE: Backend>(module: &Module<BE>, ct: &mut CKKSCiphertext<impl DataMut>)
where
    Module<BE>: VecZnxNegateInplace,
{
    ct.assert_valid("neg_inplace input");
    let ncols = ct.inner.data().cols;
    for i in 0..ncols {
        module.vec_znx_negate_inplace(ct.inner.data_mut(), i);
    }
    ct.assert_valid("neg_inplace result");
}
