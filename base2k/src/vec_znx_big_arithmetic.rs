use crate::ffi::vec_znx_big::{
    delete_vec_znx_big, new_vec_znx_big, vec_znx_big_add_small, vec_znx_big_automorphism,
    vec_znx_big_normalize_base2k, vec_znx_big_normalize_base2k_tmp_bytes, vec_znx_big_sub_small_a,
    vec_znx_bigcoeff_t,
};
use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::Free;
use crate::{Module, VecZnx, VecZnxDft};

pub struct VecZnxBig(pub *mut vec_znx_bigcoeff_t, pub usize);

impl VecZnxBig {
    pub fn as_vec_znx_dft(&mut self) -> VecZnxDft {
        VecZnxDft(self.0 as *mut vec_znx_dft_t, self.1)
    }
    pub fn limbs(&self) -> usize {
        self.1
    }
}

impl Free for VecZnxBig {
    fn free(self) {
        unsafe {
            delete_vec_znx_big(self.0);
        }
        drop(self);
    }
}

impl Module {
    // Allocates a vector Z[X]/(X^N+1) that stores not normalized values.
    pub fn new_vec_znx_big(&self, limbs: usize) -> VecZnxBig {
        unsafe { VecZnxBig(new_vec_znx_big(self.0, limbs as u64), limbs) }
    }

    // b <- b - a
    pub fn vec_znx_big_sub_small_a_inplace(&self, b: &mut VecZnxBig, a: &VecZnx) {
        let limbs: usize = a.limbs();
        assert!(
            b.limbs() >= limbs,
            "invalid c_vector: b.limbs()={} < a.limbs()={}",
            b.limbs(),
            limbs
        );
        unsafe {
            vec_znx_big_sub_small_a(
                self.0,
                b.0,
                b.limbs() as u64,
                a.as_ptr(),
                limbs as u64,
                a.n() as u64,
                b.0,
                b.limbs() as u64,
            )
        }
    }

    // c <- b - a
    pub fn vec_znx_big_sub_small_a(&self, c: &mut VecZnxBig, a: &VecZnx, b: &VecZnxBig) {
        let limbs: usize = a.limbs();
        assert!(
            b.limbs() >= limbs,
            "invalid c: b.limbs()={} < a.limbs()={}",
            b.limbs(),
            limbs
        );
        assert!(
            c.limbs() >= limbs,
            "invalid c: c.limbs()={} < a.limbs()={}",
            c.limbs(),
            limbs
        );
        unsafe {
            vec_znx_big_sub_small_a(
                self.0,
                c.0,
                c.limbs() as u64,
                a.as_ptr(),
                limbs as u64,
                a.n() as u64,
                b.0,
                b.limbs() as u64,
            )
        }
    }

    // c <- b + a
    pub fn vec_znx_big_add_small(&self, c: &mut VecZnxBig, a: &VecZnx, b: &VecZnxBig) {
        let limbs: usize = a.limbs();
        assert!(
            b.limbs() >= limbs,
            "invalid c: b.limbs()={} < a.limbs()={}",
            b.limbs(),
            limbs
        );
        assert!(
            c.limbs() >= limbs,
            "invalid c: c.limbs()={} < a.limbs()={}",
            c.limbs(),
            limbs
        );
        unsafe {
            vec_znx_big_add_small(
                self.0,
                c.0,
                limbs as u64,
                b.0,
                limbs as u64,
                a.as_ptr(),
                limbs as u64,
                a.n() as u64,
            )
        }
    }

    // b <- b + a
    pub fn vec_znx_big_add_small_inplace(&self, b: &mut VecZnxBig, a: &VecZnx) {
        let limbs: usize = a.limbs();
        assert!(
            b.limbs() >= limbs,
            "invalid c_vector: b.limbs()={} < a.limbs()={}",
            b.limbs(),
            limbs
        );
        unsafe {
            vec_znx_big_add_small(
                self.0,
                b.0,
                limbs as u64,
                b.0,
                limbs as u64,
                a.as_ptr(),
                limbs as u64,
                a.n() as u64,
            )
        }
    }

    pub fn vec_znx_big_normalize_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_big_normalize_base2k_tmp_bytes(self.0) as usize }
    }

    // b <- normalize(a)
    pub fn vec_znx_big_normalize(
        &self,
        log_base2k: usize,
        b: &mut VecZnx,
        a: &VecZnxBig,
        tmp_bytes: &mut [u8],
    ) {
        let limbs: usize = b.limbs();
        assert!(
            b.limbs() >= limbs,
            "invalid c_vector: b.limbs()={} < a.limbs()={}",
            b.limbs(),
            limbs
        );
        assert!(
            tmp_bytes.len() >= self.vec_znx_big_normalize_tmp_bytes(),
            "invalid tmp_bytes: tmp_bytes.len()={} <= self.vec_znx_big_normalize_tmp_bytes()={}",
            tmp_bytes.len(),
            self.vec_znx_big_normalize_tmp_bytes()
        );
        unsafe {
            vec_znx_big_normalize_base2k(
                self.0,
                log_base2k as u64,
                b.as_mut_ptr(),
                limbs as u64,
                b.n() as u64,
                a.0,
                limbs as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    pub fn vec_znx_big_automorphism(&self, gal_el: i64, b: &mut VecZnxBig, a: &VecZnxBig) {
        unsafe {
            vec_znx_big_automorphism(self.0, gal_el, b.0, b.limbs() as u64, a.0, a.limbs() as u64);
        }
    }

    pub fn vec_znx_big_automorphism_inplace(&self, gal_el: i64, a: &mut VecZnxBig) {
        unsafe {
            vec_znx_big_automorphism(self.0, gal_el, a.0, a.limbs() as u64, a.0, a.limbs() as u64);
        }
    }
}
