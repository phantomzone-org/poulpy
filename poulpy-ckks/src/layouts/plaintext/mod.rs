mod cst;
mod vec;

pub use cst::{CKKSConstPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx};
pub use vec::{CKKSPlaintextConversion, CKKSPlaintextVecRnx, CKKSPlaintextVecZnx};

/// Conventional alias for vector CKKS plaintexts in RNX form.
pub type CKKSPlaintextRnx<F> = CKKSPlaintextVecRnx<F>;
/// Conventional alias for vector CKKS plaintexts in ZNX form.
pub type CKKSPlaintextZnx<D> = CKKSPlaintextVecZnx<D>;
