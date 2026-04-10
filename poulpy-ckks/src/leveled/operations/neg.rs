//! CKKS ciphertext negation.
//!
//! Negates each column of the GLWE ciphertext.

use crate::layouts::ciphertext::CKKSCiphertext;
use poulpy_hal::{
    api::{VecZnxNegate, VecZnxNegateInplace},
    layouts::{Backend, DataMut, DataRef, Module},
};

impl<D: DataMut> CKKSCiphertext<D> {
    pub fn neg<BE: Backend>(&mut self, module: &Module<BE>, ct: &CKKSCiphertext<impl DataRef>)
    where
        Module<BE>: VecZnxNegate,
    {
        let ncols = ct.inner.data().cols;
        for i in 0..ncols {
            module.vec_znx_negate(self.inner.data_mut(), i, ct.inner.data(), i);
        }
        self.log_delta = ct.log_delta;
    }

    pub fn neg_inplace<BE: Backend>(&mut self, module: &Module<BE>)
    where
        Module<BE>: VecZnxNegateInplace,
    {
        let ncols = self.inner.data().cols;
        for i in 0..ncols {
            module.vec_znx_negate_inplace(self.inner.data_mut(), i);
        }
    }
}
