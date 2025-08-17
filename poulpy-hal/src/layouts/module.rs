use std::{fmt::Display, marker::PhantomData, ptr::NonNull};

use rand_distr::num_traits::Zero;

use crate::GALOISGENERATOR;

#[allow(clippy::missing_safety_doc)]
pub trait Backend: Sized {
    type ScalarBig: Copy + Zero + Display;
    type ScalarPrep: Copy + Zero + Display;
    type Handle: 'static;
    fn layout_prep_word_count() -> usize;
    fn layout_big_word_count() -> usize;
    unsafe fn destroy(handle: NonNull<Self::Handle>);
}

pub struct Module<B: Backend> {
    ptr: NonNull<B::Handle>,
    n: u64,
    _marker: PhantomData<B>,
}

impl<B: Backend> Module<B> {
    /// Construct from a raw pointer managed elsewhere.
    /// SAFETY: `ptr` must be non-null and remain valid for the lifetime of this Module.
    #[inline]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn from_raw_parts(ptr: *mut B::Handle, n: u64) -> Self {
        Self {
            ptr: NonNull::new(ptr).expect("null module ptr"),
            n,
            _marker: PhantomData,
        }
    }

    #[allow(clippy::missing_safety_doc)]
    #[inline]
    pub unsafe fn ptr(&self) -> *mut <B as Backend>::Handle {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n as usize
    }
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut B::Handle {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn log_n(&self) -> usize {
        (usize::BITS - (self.n() - 1).leading_zeros()) as _
    }

    #[inline]
    pub fn cyclotomic_order(&self) -> u64 {
        (self.n() << 1) as _
    }

    // Returns GALOISGENERATOR^|generator| * sign(generator)
    #[inline]
    pub fn galois_element(&self, generator: i64) -> i64 {
        if generator == 0 {
            return 1;
        }
        ((mod_exp_u64(GALOISGENERATOR, generator.unsigned_abs() as usize) & (self.cyclotomic_order() - 1)) as i64)
            * generator.signum()
    }

    // Returns gen^-1
    #[inline]
    pub fn galois_element_inv(&self, gal_el: i64) -> i64 {
        if gal_el == 0 {
            panic!("cannot invert 0")
        }
        ((mod_exp_u64(
            gal_el.unsigned_abs(),
            (self.cyclotomic_order() - 1) as usize,
        ) & (self.cyclotomic_order() - 1)) as i64)
            * gal_el.signum()
    }
}

impl<B: Backend> Drop for Module<B> {
    fn drop(&mut self) {
        unsafe { B::destroy(self.ptr) }
    }
}

pub fn mod_exp_u64(x: u64, e: usize) -> u64 {
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
