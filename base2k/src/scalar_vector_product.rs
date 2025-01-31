use crate::ffi::svp::{delete_svp_ppol, new_svp_ppol, svp_apply_dft, svp_ppol_t, svp_prepare};
use crate::scalar::Scalar;
use crate::{Module, VecZnx, VecZnxDft};

pub struct SvpPPol(pub *mut svp_ppol_t, pub usize);

impl SvpPPol {
    pub fn n(&self) -> usize {
        self.1
    }

    pub fn delete(self) {
        unsafe { delete_svp_ppol(self.0) };
        let _ = drop(self);
    }
}

impl Module {
    // Prepares a scalar polynomial (1 limb) for a scalar x vector product.
    // Method will panic if a.limbs() != 1.
    pub fn svp_prepare(&self, svp_ppol: &mut SvpPPol, a: &Scalar) {
        unsafe { svp_prepare(self.0, svp_ppol.0, a.as_ptr()) }
    }

    // Allocates a scalar-vector-product prepared-poly (VecZnxBig).
    pub fn svp_new_ppol(&self) -> SvpPPol {
        unsafe { SvpPPol(new_svp_ppol(self.0), self.n()) }
    }

    // Applies a scalar x vector product: res <- a (ppol) x b
    pub fn svp_apply_dft(&self, c: &mut VecZnxDft, a: &SvpPPol, b: &VecZnx) {
        let limbs: u64 = b.limbs() as u64;
        assert!(
            c.limbs() as u64 >= limbs,
            "invalid c_vector: c_vector.limbs()={} < b.limbs()={}",
            c.limbs(),
            limbs
        );
        unsafe { svp_apply_dft(self.0, c.0, limbs, a.0, b.as_ptr(), limbs, b.n() as u64) }
    }
}
