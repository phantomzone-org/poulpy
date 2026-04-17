mod cst;
mod vec;

pub use cst::{CKKSConstPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx};
pub use vec::{CKKSPlaintextConversion, CKKSPlaintextVecRnx, CKKSPlaintextVecZnx, alloc_pt_vec_znx, alloc_pt_znx};

pub type CKKSPlaintextRnx<F> = CKKSPlaintextVecRnx<F>;
pub type CKKSPlaintextZnx<D> = CKKSPlaintextVecZnx<D>;
