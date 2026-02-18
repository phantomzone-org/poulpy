use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    ptr::NonNull,
};

use bytemuck::Pod;
use rand_distr::num_traits::Zero;

use crate::{
    GALOISGENERATOR,
    api::{ModuleLogN, ModuleN},
};

#[allow(clippy::missing_safety_doc)]
pub trait Backend: Sized + Sync + Send {
    type ScalarBig: Copy + Zero + Display + Debug + Pod;
    type ScalarPrep: Copy + Zero + Display + Debug + Pod;
    type Handle: 'static;
    fn layout_prep_word_count() -> usize;
    fn layout_big_word_count() -> usize;
    unsafe fn destroy(handle: NonNull<Self::Handle>);
}

pub struct Module<B: Backend> {
    ptr: Option<NonNull<B::Handle>>,
    n: u64,
    _marker: PhantomData<B>,
}

unsafe impl<B: Backend> Sync for Module<B> {}
unsafe impl<B: Backend> Send for Module<B> {}

impl<B: Backend> Module<B> {
    /// Creates a marker module with no backend handle.
    /// Operations requiring a backend handle will panic.
    #[inline]
    pub fn new_marker(n: u64) -> Self {
        assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
        Self {
            ptr: None,
            n,
            _marker: PhantomData,
        }
    }

    #[allow(clippy::missing_safety_doc)]
    #[inline]
    pub unsafe fn from_nonnull(ptr: NonNull<B::Handle>, n: u64) -> Self {
        assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
        Self {
            ptr: Some(ptr),
            n,
            _marker: PhantomData,
        }
    }

    /// Construct from a raw pointer managed elsewhere.
    /// SAFETY: `ptr` must be non-null and remain valid for the lifetime of this Module.
    #[inline]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn from_raw_parts(ptr: *mut B::Handle, n: u64) -> Self {
        assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
        Self {
            ptr: Some(NonNull::new(ptr).expect("null module ptr")),
            n,
            _marker: PhantomData,
        }
    }

    /// Returns the raw pointer to the backend handle.
    ///
    /// # Panics
    /// Panics if this is a marker module (created via `new_marker`).
    #[allow(clippy::missing_safety_doc)]
    #[inline]
    pub unsafe fn ptr(&self) -> *mut <B as Backend>::Handle {
        self.ptr.expect("called ptr() on a marker module (no backend handle)").as_ptr()
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n as usize
    }

    /// Returns the raw pointer to the backend handle.
    ///
    /// # Panics
    /// Panics if this is a marker module (created via `new_marker`).
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut B::Handle {
        self.ptr.expect("called as_mut_ptr() on a marker module (no backend handle)").as_ptr()
    }

    /// Returns true if this module has a backend handle (not a marker).
    #[inline]
    pub fn has_handle(&self) -> bool {
        self.ptr.is_some()
    }

    #[inline]
    pub fn log_n(&self) -> usize {
        (usize::BITS - (self.n() - 1).leading_zeros()) as _
    }
}

pub trait CyclotomicOrder
where
    Self: ModuleN,
{
    fn cyclotomic_order(&self) -> i64 {
        (self.n() << 1) as _
    }
}

impl<BE: Backend> ModuleLogN for Module<BE> where Self: ModuleN {}

impl<BE: Backend> CyclotomicOrder for Module<BE> where Self: ModuleN {}

#[inline(always)]
pub fn galois_element(generator: i64, cyclotomic_order: i64) -> i64 {
    debug_assert!(
        cyclotomic_order > 0 && (cyclotomic_order as u64).is_power_of_two(),
        "cyclotomic_order must be a power of two, got {cyclotomic_order}"
    );

    if generator == 0 {
        return 1;
    }

    let g_exp: u64 = mod_exp_u64(GALOISGENERATOR, generator.unsigned_abs() as usize) & (cyclotomic_order - 1) as u64;
    g_exp as i64 * generator.signum()
}

pub trait GaloisElement
where
    Self: CyclotomicOrder,
{
    // Returns GALOISGENERATOR^|generator| * sign(generator)
    fn galois_element(&self, generator: i64) -> i64 {
        galois_element(generator, self.cyclotomic_order())
    }

    // Returns gen^-1
    fn galois_element_inv(&self, gal_el: i64) -> i64 {
        if gal_el == 0 {
            panic!("cannot invert 0")
        }

        let g_exp: u64 =
            mod_exp_u64(gal_el.unsigned_abs(), (self.cyclotomic_order() - 1) as usize) & (self.cyclotomic_order() - 1) as u64;
        g_exp as i64 * gal_el.signum()
    }
}

impl<BE: Backend> GaloisElement for Module<BE> where Self: CyclotomicOrder {}

impl<B: Backend> Drop for Module<B> {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr {
            unsafe { B::destroy(ptr) }
        }
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
