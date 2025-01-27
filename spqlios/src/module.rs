use crate::bindings::*;
use crate::poly::Poly;
use crate::scalar::Scalar;

pub type MODULETYPE = u8;
pub const FFT64: u8 = 0;
pub const NTT120: u8 = 1;

pub struct Module(*mut MODULE);

impl Module {
    // Instantiates a new module.
    pub fn new<const MODULETYPE: MODULETYPE>(n: usize) -> Self {
        unsafe {
            let m: *mut module_info_t = new_module_info(n as u64, MODULETYPE as u32);
            if m.is_null() {
                panic!("Failed to create module.");
            }
            Self(m)
        }
    }

    // Prepares a scalar polynomial (1 limb) for a scalar x vector product.
    // Method will panic if a.limbs() != 1.
    pub fn svp_prepare(&self, svp_ppol: &mut SVPPOL, a: &Scalar) {
        unsafe { svp_prepare(self.0, svp_ppol.0, a.as_ptr()) }
    }

    // Allocates a scalar-vector-product prepared-poly (SVPPOL).
    pub fn new_svp_ppol(&self) -> SVPPOL {
        unsafe { SVPPOL(new_svp_ppol(self.0)) }
    }

    // Allocates a vector Z[X]/(X^N+1) that stores normalized in the DFT space.
    pub fn new_vec_znx_dft(&self, limbs: usize) -> VECZNXDFT {
        unsafe { VECZNXDFT(new_vec_znx_dft(self.0, limbs as u64), limbs) }
    }

    // Allocates a vector Z[X]/(X^N+1) that stores not normalized values.
    pub fn new_vec_znx_big(&self, limbs: usize) -> VECZNXBIG {
        unsafe { VECZNXBIG(new_vec_znx_big(self.0, limbs as u64), limbs) }
    }

    // Applies a scalar x vector product: res <- a (ppol) x b
    pub fn svp_apply_dft(&self, c: &mut VECZNXDFT, a: &SVPPOL, b: &Poly) {
        let limbs: u64 = b.limbs() as u64;
        assert!(
            c.limbs() as u64 >= limbs,
            "invalid c_vector: c_vector.limbs()={} < b.limbs()={}",
            c.limbs(),
            limbs
        );
        unsafe { svp_apply_dft(self.0, c.0, limbs, a.0, b.as_ptr(), limbs, b.n() as u64) }
    }

    // b <- IDFT(a), uses a as scratch space.
    pub fn vec_znx_idft_tmp_a(&self, b: &mut VECZNXBIG, a: &mut VECZNXDFT, a_limbs: usize) {
        assert!(
            b.limbs() >= a_limbs,
            "invalid c_vector: b_vector.limbs()={} < a_limbs={}",
            b.limbs(),
            a_limbs
        );
        unsafe { vec_znx_idft_tmp_a(self.0, b.0, a_limbs as u64, a.0, a_limbs as u64) }
    }

    // Returns the size of the scratch space for [vec_znx_idft].
    pub fn vec_znx_idft_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_idft_tmp_bytes(self.0) as usize }
    }

    // b <- IDFT(a), scratch space size obtained with [vec_znx_idft_tmp_bytes].
    pub fn vec_znx_idft(
        &self,
        b_vector: &mut VECZNXBIG,
        a_vector: &mut VECZNXDFT,
        a_limbs: usize,
        tmp_bytes: &mut [u8],
    ) {
        assert!(
            b_vector.limbs() >= a_limbs,
            "invalid c_vector: b_vector.limbs()={} < a_limbs={}",
            b_vector.limbs(),
            a_limbs
        );
        assert!(
            a_vector.limbs() >= a_limbs,
            "invalid c_vector: c_vector.limbs()={} < a_limbs={}",
            a_vector.limbs(),
            a_limbs
        );
        assert!(
            tmp_bytes.len() <= self.vec_znx_idft_tmp_bytes(),
            "invalid tmp_bytes: tmp_bytes.len()={} < self.vec_znx_idft_tmp_bytes()={}",
            tmp_bytes.len(),
            self.vec_znx_idft_tmp_bytes()
        );
        unsafe {
            vec_znx_idft(
                self.0,
                b_vector.0,
                a_limbs as u64,
                a_vector.0,
                a_limbs as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    // c <- b - a
    pub fn vec_znx_big_sub_small_a(&self, c: &mut VECZNXBIG, a: &Poly, b: &VECZNXBIG) {
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

    // b <- b - a
    pub fn vec_znx_big_sub_small_a_inplace(&self, b: &mut VECZNXBIG, a: &Poly) {
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

    // c <- b + a
    pub fn vec_znx_big_add_small(&self, c: &mut VECZNXBIG, a: &Poly, b: &VECZNXBIG) {
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
    pub fn vec_znx_big_add_small_inplace(&self, b: &mut VECZNXBIG, a: &Poly) {
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
    pub fn vec_znx_big_normalize(&self, b: &mut Poly, a: &VECZNXBIG, tmp_bytes: &mut [u8]) {
        let limbs: usize = b.limbs();
        assert!(
            b.limbs() >= limbs,
            "invalid c_vector: b.limbs()={} < a.limbs()={}",
            b.limbs(),
            limbs
        );
        assert!(
            tmp_bytes.len() <= self.vec_znx_big_normalize_tmp_bytes(),
            "invalid tmp_bytes: tmp_bytes.len()={} <= self.vec_znx_big_normalize_tmp_bytes()={}",
            tmp_bytes.len(),
            self.vec_znx_big_normalize_tmp_bytes()
        );
        unsafe {
            vec_znx_big_normalize_base2k(
                self.0,
                b.log_base2k as u64,
                b.as_mut_ptr(),
                limbs as u64,
                b.n() as u64,
                a.0,
                limbs as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }
}

pub struct SVPPOL(*mut svp_ppol_t);
pub struct VECZNXDFT(*mut vec_znx_dft_t, usize);
pub struct VECZNXBIG(*mut vec_znx_bigcoeff_t, usize);

impl VECZNXBIG {
    pub fn as_vec_znx_dft(&mut self) -> VECZNXDFT {
        VECZNXDFT(self.0 as *mut vec_znx_dft_t, self.1)
    }
    pub fn limbs(&self) -> usize {
        self.1
    }
}

impl VECZNXDFT {
    pub fn as_vec_znx_big(&mut self) -> VECZNXBIG {
        VECZNXBIG(self.0 as *mut vec_znx_bigcoeff_t, self.1)
    }
    pub fn limbs(&self) -> usize {
        self.1
    }
}
