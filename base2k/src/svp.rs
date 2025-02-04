use crate::ffi::svp::{delete_svp_ppol, new_svp_ppol, svp_apply_dft, svp_ppol_t, svp_prepare};
use crate::scalar::Scalar;
use crate::{Free, Module, VecZnx, VecZnxDft};

pub struct SvpPPol(pub *mut svp_ppol_t, pub usize);

/// A prepared [crate::Scalar] for [ScalarVectorProduct::svp_apply_dft].
/// An [SvpPPol] an be seen as a [VecZnxDft] of one limb.
/// The backend array of an [SvpPPol] is allocated in C and must be freed manually.
impl SvpPPol {
    /// Returns the ring degree of the [SvpPPol].
    pub fn n(&self) -> usize {
        self.1
    }

    /// Returns the number of limbs of the [SvpPPol], which is always 1.
    pub fn limbs(&self) -> usize {
        1
    }
}

impl Free for SvpPPol {
    fn free(self) {
        unsafe { delete_svp_ppol(self.0) };
        let _ = drop(self);
    }
}

pub trait ScalarVectorProduct {
    /// Prepares a [crate::Scalar] for a [ScalarVectorProduct::svp_apply_dft].
    fn svp_prepare(&self, svp_ppol: &mut SvpPPol, a: &Scalar);

    /// Allocates a new [SvpPPol].
    fn svp_new_ppol(&self) -> SvpPPol;

    /// Applies the [SvpPPol] x [VecZnxDft] product, where each limb of
    /// the [VecZnxDft] is multiplied with [SvpPPol].
    fn svp_apply_dft(&self, c: &mut VecZnxDft, a: &SvpPPol, b: &VecZnx);
}

impl Module {
    pub fn svp_prepare(&self, svp_ppol: &mut SvpPPol, a: &Scalar) {
        unsafe { svp_prepare(self.0, svp_ppol.0, a.as_ptr()) }
    }

    pub fn svp_new_ppol(&self) -> SvpPPol {
        unsafe { SvpPPol(new_svp_ppol(self.0), self.n()) }
    }

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
