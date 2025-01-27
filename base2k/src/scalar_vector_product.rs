use crate::bindings::{new_svp_ppol, svp_apply_dft, svp_prepare};
use crate::module::{Module, SVPPOL, VECZNXDFT};
use crate::scalar::Scalar;
use crate::vector::Vector;

impl Module {
    // Prepares a scalar polynomial (1 limb) for a scalar x vector product.
    // Method will panic if a.limbs() != 1.
    pub fn svp_prepare(&self, svp_ppol: &mut SVPPOL, a: &Scalar) {
        unsafe { svp_prepare(self.0, svp_ppol.0, a.as_ptr()) }
    }

    // Allocates a scalar-vector-product prepared-poly (SVPPOL).
    pub fn svp_new_ppol(&self) -> SVPPOL {
        unsafe { SVPPOL(new_svp_ppol(self.0)) }
    }

    // Applies a scalar x vector product: res <- a (ppol) x b
    pub fn svp_apply_dft(&self, c: &mut VECZNXDFT, a: &SVPPOL, b: &Vector) {
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
