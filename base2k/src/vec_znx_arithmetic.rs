use crate::bindings::{vec_znx_add, vec_znx_automorphism, vec_znx_rotate, vec_znx_sub};
use crate::{Module, VecZnx};

impl Module {
    // c <- a + b
    pub fn vec_znx_add(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx) {
        unsafe {
            vec_znx_add(
                self.0,
                c.as_mut_ptr(),
                c.limbs() as u64,
                c.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                b.as_ptr(),
                b.limbs() as u64,
                b.n() as u64,
            )
        }
    }

    // b <- a + b
    pub fn vec_znx_add_inplace(&self, b: &mut VecZnx, a: &VecZnx) {
        unsafe {
            vec_znx_add(
                self.0,
                b.as_mut_ptr(),
                b.limbs() as u64,
                b.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                b.as_ptr(),
                b.limbs() as u64,
                b.n() as u64,
            )
        }
    }

    // c <- a + b
    pub fn vec_znx_sub(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx) {
        unsafe {
            vec_znx_sub(
                self.0,
                c.as_mut_ptr(),
                c.limbs() as u64,
                c.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                b.as_ptr(),
                b.limbs() as u64,
                b.n() as u64,
            )
        }
    }

    // b <- a + b
    pub fn vec_znx_sub_inplace(&self, b: &mut VecZnx, a: &VecZnx) {
        unsafe {
            vec_znx_sub(
                self.0,
                b.as_mut_ptr(),
                b.limbs() as u64,
                b.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                b.as_ptr(),
                b.limbs() as u64,
                b.n() as u64,
            )
        }
    }

    pub fn vec_znx_rotate(&self, k: i64, a: &mut VecZnx, b: &VecZnx) {
        unsafe {
            vec_znx_rotate(
                self.0,
                k,
                a.as_mut_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                b.as_ptr(),
                b.limbs() as u64,
                b.n() as u64,
            )
        }
    }

    pub fn vec_znx_rotate_inplace(&self, k: i64, a: &mut VecZnx) {
        unsafe {
            vec_znx_rotate(
                self.0,
                k,
                a.as_mut_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
            )
        }
    }

    // b <- a(X^gal_el)
    pub fn vec_znx_automorphism(&self, gal_el: i64, b: &mut VecZnx, a: &VecZnx) {
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

    // a <- a(X^gal_el)
    pub fn vec_znx_automorphism_inplace(&self, gal_el: i64, a: &mut VecZnx) {
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
