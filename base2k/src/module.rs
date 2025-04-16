use crate::ffi::module::{delete_module_info, module_info_t, new_module_info, MODULE};
use crate::GALOISGENERATOR;

#[derive(Copy, Clone)]
#[repr(u8)]
pub enum BACKEND {
    FFT64,
    NTT120,
}

pub struct Module {
    pub ptr: *mut MODULE,
    pub n: usize,
    pub backend: BACKEND,
}

impl Module {
    // Instantiates a new module.
    pub fn new(n: usize, module_type: BACKEND) -> Self {
        unsafe {
            let module_type_u32: u32;
            match module_type {
                BACKEND::FFT64 => module_type_u32 = 0,
                BACKEND::NTT120 => module_type_u32 = 1,
            }
            let m: *mut module_info_t = new_module_info(n as u64, module_type_u32);
            if m.is_null() {
                panic!("Failed to create module.");
            }
            Self {
                ptr: m,
                n: n,
                backend: module_type,
            }
        }
    }

    pub fn backend(&self) -> BACKEND {
        self.backend
    }

    pub fn n(&self) -> usize {
        self.n
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

    pub fn free(self) {
        unsafe { delete_module_info(self.ptr) }
        drop(self);
    }
}
