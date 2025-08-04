use crate::{
    hal::{api::ZnxView, layouts::VmpPMat},
    implementation::cpu_spqlios::module_ntt120::NTT120,
};

impl<D: AsRef<[u8]>> ZnxView for VmpPMat<D, NTT120> {
    type Scalar = i64;
}
