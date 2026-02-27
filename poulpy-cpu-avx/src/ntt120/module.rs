//! Backend handle and module initialisation for [`NTT120Avx`](super::NTT120Avx).
//!
//! This module defines:
//!
//! - [`NTT120AvxHandle`]: the opaque handle stored inside a `Module<NTT120Avx>`,
//!   holding precomputed NTT and iNTT twiddle-factor tables and multiply-accumulate metadata.
//! - The [`Backend`] trait implementation, which defines scalar types and the
//!   handle destruction path.
//! - The [`ModuleNewImpl`] implementation, which allocates the handle on the heap,
//!   verifies AVX2 availability at runtime, and transfers ownership to the `Module`.
//! - The [`NttHandleProvider`] impl for [`NTT120AvxHandle`], wiring the handle into
//!   the blanket `NttModuleHandle` impl provided by `poulpy-hal`.

use std::ptr::NonNull;

use poulpy_hal::{
    layouts::{Backend, Module},
    oep::ModuleNewImpl,
    reference::ntt120::{
        mat_vec::{BbbMeta, BbcMeta},
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
        types::Q120bScalar,
        vec_znx_dft::NttHandleProvider,
    },
};

use super::NTT120Avx;

/// Opaque handle for the [`NTT120Avx`](super::NTT120Avx) backend.
///
/// Holds precomputed twiddle-factor tables for the forward NTT and inverse NTT
/// of size `n`, and the lazy-accumulation metadata for `q120b × q120c` and
/// `q120b × q120b` products.
///
/// This struct is heap-allocated during module creation and freed when the
/// `Module<NTT120Avx>` is dropped (via [`Backend::destroy`]).
#[repr(C)]
pub struct NTT120AvxHandle {
    table_ntt: NttTable<Primes30>,
    table_intt: NttTableInv<Primes30>,
    meta_bbc: BbcMeta<Primes30>,
    meta_bbb: BbbMeta<Primes30>,
}

impl Backend for NTT120Avx {
    type ScalarPrep = Q120bScalar;
    type ScalarBig = i128;
    type Handle = NTT120AvxHandle;

    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe {
            drop(Box::from_raw(handle.as_ptr()));
        }
    }
}

/// # Safety
///
/// The returned `Module` owns the heap-allocated `NTT120AvxHandle`.
/// `n` must be a power of two >= 2 (asserted by `Module::from_nonnull`).
/// The NTT tables are built for dimension `n`.
///
/// # Panics
///
/// Panics if the runtime CPU does not support the AVX2 instruction set.
unsafe impl ModuleNewImpl<Self> for NTT120Avx {
    fn new_impl(n: u64) -> Module<Self> {
        if !std::arch::is_x86_feature_detected!("avx2") {
            panic!("arch must support avx2")
        }

        let handle = NTT120AvxHandle {
            table_ntt: NttTable::new(n as usize),
            table_intt: NttTableInv::new(n as usize),
            meta_bbc: BbcMeta::new(),
            meta_bbb: BbbMeta::new(),
        };
        let ptr: NonNull<NTT120AvxHandle> = NonNull::from(Box::leak(Box::new(handle)));
        unsafe { Module::from_nonnull(ptr, n) }
    }
}

/// # Safety
///
/// The returned references are valid for the lifetime of `&self`.
/// All fields are fully initialised in [`NTT120Avx::new_impl`].
unsafe impl NttHandleProvider for NTT120AvxHandle {
    fn get_ntt_table(&self) -> &NttTable<Primes30> {
        &self.table_ntt
    }

    fn get_intt_table(&self) -> &NttTableInv<Primes30> {
        &self.table_intt
    }

    fn get_bbc_meta(&self) -> &BbcMeta<Primes30> {
        &self.meta_bbc
    }

    fn get_bbb_meta(&self) -> &BbbMeta<Primes30> {
        &self.meta_bbb
    }
}
