use crate::GALOISGENERATOR;
use crate::ffi::module::{MODULE, delete_module_info, module_info_t, new_module_info};
use std::fmt;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BACKEND {
    FFT64,
    NTT120,
}

pub type ModuleKey = (BACKEND, u64);

pub enum ModuleInfos{
    FFT64(Module<FFT64>),
    NTT120(Module<NTT120>),
}

impl fmt::Debug for ModuleInfos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModuleInfos::FFT64(_) => write!(f, "ModuleInfos::FFT(<Module<FFT64>>"),
            ModuleInfos::NTT120(_) => write!(f, "ModuleInfos::NTT(<Module<NTT120>>"),
        }
    }
}


impl ModuleInfos{
    pub fn fft64(n: usize) -> Self{
        ModuleInfos::FFT64(Module::<FFT64>::new(n))
    }

    pub fn ntt120(n: usize) -> Self{
        ModuleInfos::FFT64(Module::<FFT64>::new(n))
    }
}


pub trait Backend: Sized {
    const KIND: BACKEND;
    fn module_type() -> u32;
    fn extract(info: &ModuleInfos) -> Option<&Module<Self>>;
}

pub struct FFT64;
pub struct NTT120;

impl Backend for FFT64 {
    const KIND: BACKEND = BACKEND::FFT64;
    fn module_type() -> u32 {
        0
    }

    fn extract(info: &ModuleInfos) -> Option<&Module<Self>> {
        match info {
            ModuleInfos::FFT64(m) => Some(m),
            _ => None,
        }
    }
}

impl Backend for NTT120 {
    const KIND: BACKEND = BACKEND::NTT120;
    fn module_type() -> u32 {
        1
    }

    fn extract(info: &ModuleInfos) -> Option<&Module<Self>> {
        match info {
            ModuleInfos::NTT120(m) => Some(m),
            _ => None,
        }
    }
}

unsafe impl<B: Backend> Send for Module<B> {}
unsafe impl<B: Backend> Sync for Module<B> {}

pub struct Module<B: Backend> {
    pub ptr: *const MODULE,
    n: usize,
    _marker: PhantomData<B>,
}

impl<B: Backend> Module<B> {
    // Instantiates a new module.
    pub fn new(n: usize) -> Self {
        unsafe {
            let m: *const module_info_t = new_module_info(n as u64, B::module_type());
            if m.is_null() {
                panic!("Failed to create module.");
            }
            Self {
                ptr: m,
                n: n,
                _marker: PhantomData,
            }
        }
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

    // Returns GALOISGENERATOR^|generator| * sign(generator)
    pub fn galois_element(&self, generator: i64) -> i64 {
        if generator == 0 {
            return 1;
        }
        ((mod_exp_u64(GALOISGENERATOR, generator.abs() as usize) & (self.cyclotomic_order() - 1)) as i64) * generator.signum()
    }

    // Returns gen^-1
    pub fn galois_element_inv(&self, gal_el: i64) -> i64 {
        if gal_el == 0 {
            panic!("cannot invert 0")
        }
        ((mod_exp_u64(gal_el.abs() as u64, (self.cyclotomic_order() - 1) as usize) & (self.cyclotomic_order() - 1)) as i64)
            * gal_el.signum()
    }
}

impl<B: Backend> Drop for Module<B> {
    fn drop(&mut self) {
        unsafe { delete_module_info(self.ptr) }
    }
}

fn mod_exp_u64(x: u64, e: usize) -> u64 {
    let mut y: u64 = 1;
    let mut x_pow: u64 = x;
    let mut exp = e;
    while exp > 0 {
        if exp & 1 == 1 {
            y = y.wrapping_mul(x_pow);
        }
        x_pow = x_pow.wrapping_mul(x_pow);
        exp >>= 1;
    }
    y
}
