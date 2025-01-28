use crate::bindings::{
    module_info_t, new_module_info, svp_ppol_t, vec_znx_bigcoeff_t, vec_znx_dft_t, MODULE,
};

use crate::GALOISGENERATOR;

pub type MODULETYPE = u8;
pub const FFT64: u8 = 0;
pub const NTT120: u8 = 1;

pub struct Module(pub *mut MODULE, pub usize);

impl Module {
    // Instantiates a new module.
    pub fn new<const MODULETYPE: MODULETYPE>(n: usize) -> Self {
        unsafe {
            let m: *mut module_info_t = new_module_info(n as u64, MODULETYPE as u32);
            if m.is_null() {
                panic!("Failed to create module.");
            }
            Self(m, n)
        }
    }

    pub fn n(&self) -> usize {
        self.1
    }

    pub fn log_n(&self) -> usize {
        (usize::BITS - (self.n() - 1).leading_zeros()) as _
    }

    pub fn cyclotomic_order(&self) -> u64 {
        (self.n() << 1) as _
    }

    // GALOISGENERATOR^|gen| * sign(gen)
    pub fn galois_element(&self, gen: i64) -> i64 {
        if gen == 0 {
            return 1;
        }

        let mut gal_el: u64 = 1;
        let mut gen_1_pow: u64 = GALOISGENERATOR;
        let mut e: usize = gen.abs() as usize;
        while e > 0 {
            if e & 1 == 1 {
                gal_el = gal_el.wrapping_mul(gen_1_pow);
            }

            gen_1_pow = gen_1_pow.wrapping_mul(gen_1_pow);
            e >>= 1;
        }

        gal_el &= self.cyclotomic_order() - 1;

        (gal_el as i64) * gen.signum()
    }
}

pub struct SvpPPol(pub *mut svp_ppol_t);

pub struct VecZnxBig(pub *mut vec_znx_bigcoeff_t, pub usize);

impl VecZnxBig {
    pub fn as_vec_znx_dft(&mut self) -> VecZnxDft {
        VecZnxDft(self.0 as *mut vec_znx_dft_t, self.1)
    }
    pub fn limbs(&self) -> usize {
        self.1
    }
}

pub struct VecZnxDft(pub *mut vec_znx_dft_t, pub usize);

impl VecZnxDft {
    pub fn as_vec_znx_big(&mut self) -> VecZnxBig {
        VecZnxBig(self.0 as *mut vec_znx_bigcoeff_t, self.1)
    }
    pub fn limbs(&self) -> usize {
        self.1
    }
}
