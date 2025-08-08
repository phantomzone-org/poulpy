use crate::{
    hal::{
        api::ZnxView,
        layouts::{DataRef, VmpPMat},
    },
    implementation::cpu_spqlios::module_ntt120::NTT120,
};

impl<D: DataRef> ZnxView for VmpPMat<D, NTT120> {
    type Scalar = i64;
}
