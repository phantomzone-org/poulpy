//! CKKS ciphertext negation.

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
    res.log_delta = ct.log_delta;
    let ncols = ct.inner.data().cols;
    for i in 0..ncols {
        module.vec_znx_negate(res.inner.data_mut(), i, ct.inner.data(), i);
    }
}

/// Computes `ct = -ct` in place.
pub fn neg_inplace<BE: Backend>(module: &Module<BE>, ct: &mut CKKSCiphertext<impl DataMut>)
where
    Module<BE>: VecZnxNegateInplace,
{
    let ncols = ct.inner.data().cols;
    for i in 0..ncols {
        module.vec_znx_negate_inplace(ct.inner.data_mut(), i);
    }
}
