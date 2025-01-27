use crate::bindings::vec_znx_automorphism;
use crate::module::Module;
use crate::vector::Vector;

impl Module {
    pub fn vec_znx_automorphism(&self, gal_el: i64, b: &mut Vector, a: &Vector) {
        unsafe {
            vec_znx_automorphism(
                self.0,
                gal_el,
                b.as_mut_ptr(),
                b.limbs() as u64,
                b.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
            );
        }
    }

    pub fn vec_znx_automorphism_inplace(&self, gal_el: i64, a: &mut Vector) {
        unsafe {
            vec_znx_automorphism(
                self.0,
                gal_el,
                a.as_mut_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
            );
        }
    }
}
