use crate::bindings::{
    module_info_t, new_module_info, svp_ppol_t, vec_znx_bigcoeff_t, vec_znx_dft_t, MODULE,
};

pub type MODULETYPE = u8;
pub const FFT64: u8 = 0;
pub const NTT120: u8 = 1;

pub struct Module(pub *mut MODULE);

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
}

pub struct SVPPOL(pub *mut svp_ppol_t);

pub struct VECZNXBIG(pub *mut vec_znx_bigcoeff_t, pub usize);

// Stores a vector of
impl VECZNXBIG {
    pub fn as_vec_znx_dft(&mut self) -> VECZNXDFT {
        VECZNXDFT(self.0 as *mut vec_znx_dft_t, self.1)
    }
    pub fn limbs(&self) -> usize {
        self.1
    }
}

pub struct VECZNXDFT(pub *mut vec_znx_dft_t, pub usize);

impl VECZNXDFT {
    pub fn as_vec_znx_big(&mut self) -> VECZNXBIG {
        VECZNXBIG(self.0 as *mut vec_znx_bigcoeff_t, self.1)
    }
    pub fn limbs(&self) -> usize {
        self.1
    }
}
