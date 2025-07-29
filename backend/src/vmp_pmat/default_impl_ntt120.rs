use crate::{NTT120, VmpPMat, ZnxView};

impl<D: AsRef<[u8]>> ZnxView for VmpPMat<D, NTT120> {
    type Scalar = i64;
}
