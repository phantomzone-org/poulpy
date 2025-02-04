use crate::ffi::svp;
use crate::ffi::vec_znx_big;
use crate::ffi::vec_znx_dft;
use crate::ffi::vmp;
use crate::{SvpPPol, VecZnxBig, VecZnxDft, VmpPMat};

/// This trait should be implemented by structs that point to
/// memory allocated through C.
pub trait Free {
    // Frees the memory and self destructs.
    fn free(self);
}

impl Free for VmpPMat {
    /// Frees the C allocated memory of the [VmpPMat] and self destructs the struct.
    fn free(self) {
        unsafe { vmp::delete_vmp_pmat(self.data) };
        drop(self);
    }
}

impl Free for VecZnxDft {
    fn free(self) {
        unsafe { vec_znx_dft::delete_vec_znx_dft(self.0) };
        drop(self);
    }
}

impl Free for VecZnxBig {
    fn free(self) {
        unsafe {
            vec_znx_big::delete_vec_znx_big(self.0);
        }
        drop(self);
    }
}

impl Free for SvpPPol {
    fn free(self) {
        unsafe { svp::delete_svp_ppol(self.0) };
        let _ = drop(self);
    }
}
