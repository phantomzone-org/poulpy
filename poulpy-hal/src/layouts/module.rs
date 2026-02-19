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

/// Core trait that every backend (CPU, GPU, FPGA, ...) must implement.
///
/// Defines the scalar types used for DFT-domain (`ScalarPrep`) and
/// extended-precision (`ScalarBig`) representations, as well as the
/// opaque `Handle` type that holds backend-specific precomputed state
/// (e.g. FFT twiddle factors).
///
/// # Safety
///
/// [`destroy`](Backend::destroy) is called during [`Module`] drop and must
/// correctly deallocate the handle without double-free.
#[allow(clippy::missing_safety_doc)]
pub trait Backend: Sized + Sync + Send {
    /// Scalar type for extended-precision (big) polynomial representations.
    type ScalarBig: Copy + Zero + Display + Debug + Pod;
    /// Scalar type for DFT-domain (prepared) polynomial representations.
    type ScalarPrep: Copy + Zero + Display + Debug + Pod;
    /// Opaque backend handle type (e.g. precomputed FFT twiddle factors).
    type Handle: 'static;
    /// Number of `ScalarPrep` words per ring element in DFT representation.
    fn layout_prep_word_count() -> usize;
    /// Number of `ScalarBig` words per ring element in big representation.
    fn layout_big_word_count() -> usize;
    /// Deallocates a backend handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid, non-dangling pointer that was previously
    /// returned by the backend's allocation routine. Must not be called
    /// more than once on the same handle.
    unsafe fn destroy(handle: NonNull<Self::Handle>);
}

/// Primary entry point for all polynomial operations over `Z[X]/(X^N + 1)`.
///
/// A `Module` pairs a ring degree `N` (always a power of two) with an
/// optional backend-specific handle that holds precomputed state. All
/// [`api`](crate::api) trait methods are dispatched through this type.
///
/// A *marker* module (created with [`Module::new_marker`]) has no backend
/// handle and can only be used for operations that do not require one
/// (e.g. metadata queries). Operations that need the handle will panic
/// on a marker module.
///
/// The module **owns** its handle; dropping the `Module` calls
/// [`Backend::destroy`] if a handle is present.
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

    /// Creates a module from a [`NonNull`] backend handle.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid, fully initialized backend handle whose
    /// lifetime is transferred to this `Module` (it will be destroyed on drop).
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

    /// Returns the ring degree `N`.
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

    /// Returns `log2(N)`.
    #[inline]
    pub fn log_n(&self) -> usize {
        (usize::BITS - (self.n() - 1).leading_zeros()) as _
    }
}

/// Returns the cyclotomic order `2N` for the ring `Z[X]/(X^N + 1)`.
pub trait CyclotomicOrder
where
    Self: ModuleN,
{
    /// Returns `2N`, the order of the cyclotomic polynomial `X^N + 1`.
    fn cyclotomic_order(&self) -> i64 {
        (self.n() << 1) as _
    }
}

impl<BE: Backend> ModuleLogN for Module<BE> where Self: ModuleN {}

impl<BE: Backend> CyclotomicOrder for Module<BE> where Self: ModuleN {}

/// Computes [`GALOISGENERATOR`]`^|generator| * sign(generator) mod cyclotomic_order`.
///
/// Returns `1` when `generator == 0`.
///
/// # Panics (debug)
///
/// Debug-asserts that `cyclotomic_order` is a positive power of two.
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

/// Galois group operations on the cyclotomic ring `Z[X]/(X^N + 1)`.
///
/// The Galois group `(Z/2NZ)*` acts on polynomials via the automorphisms
/// `X -> X^k` for odd `k`. This trait provides methods to compute
/// Galois elements and their inverses from a signed generator exponent.
pub trait GaloisElement
where
    Self: CyclotomicOrder,
{
    /// Returns [`GALOISGENERATOR`]`^|generator| * sign(generator) mod 2N`.
    fn galois_element(&self, generator: i64) -> i64 {
        galois_element(generator, self.cyclotomic_order())
    }

    /// Returns the inverse of `gal_el` in the Galois group `(Z/2NZ)*`.
    ///
    /// # Panics
    ///
    /// Panics if `gal_el == 0`.
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

/// Computes `x^e mod 2^64` using square-and-multiply with wrapping arithmetic.
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
