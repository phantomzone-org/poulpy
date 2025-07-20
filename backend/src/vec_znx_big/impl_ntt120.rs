use crate::{NTT120, VecZnxBig, ZnxView};

impl<D: AsRef<[u8]>> ZnxView for VecZnxBig<D, NTT120> {
    type Scalar = i128;
}
