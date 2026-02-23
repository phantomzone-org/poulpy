//! Backend handle and module initialisation for [`NTT120Ref`](crate::NTT120Ref).
//!
//! This module defines:
//!
//! - [`NTT120RefHandle`]: the opaque handle stored inside a `Module<NTT120Ref>`,
//!   holding precomputed NTT and iNTT twiddle-factor tables and multiply-accumulate metadata.
//! - The [`Backend`] trait implementation, which defines scalar types and the
//!   handle destruction path.
//! - The [`ModuleNewImpl`] implementation, which allocates the handle on the heap
//!   and transfers ownership to the `Module`.
//! - The [`NttHandleProvider`] impl for [`NTT120RefHandle`], wiring the handle into
//!   the blanket `NttModuleHandle` impl provided by `poulpy-hal`.

use std::ptr::NonNull;

use poulpy_hal::{
    layouts::{Backend, Module},
    oep::ModuleNewImpl,
    reference::ntt120::{
        mat_vec::BbcMeta,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
        types::Q120bScalar,
        vec_znx_dft::NttHandleProvider,
    },
};

use crate::NTT120Ref;

/// Opaque handle for the [`NTT120Ref`](crate::NTT120Ref) backend.
///
/// Holds precomputed twiddle-factor tables for the forward NTT and inverse NTT
/// of size `n`, and the lazy-accumulation metadata for `q120b × q120c` products.
///
/// This struct is heap-allocated during module creation and freed when the
/// `Module<NTT120Ref>` is dropped (via [`Backend::destroy`]).
#[repr(C)]
pub struct NTT120RefHandle {
    table_ntt: NttTable<Primes30>,
    table_intt: NttTableInv<Primes30>,
    meta_bbc: BbcMeta<Primes30>,
}

impl Backend for NTT120Ref {
    type ScalarPrep = Q120bScalar;
    type ScalarBig = i128;
    type Handle = NTT120RefHandle;

    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe {
            drop(Box::from_raw(handle.as_ptr()));
        }
    }

    fn layout_big_word_count() -> usize {
        1
    }

    fn layout_prep_word_count() -> usize {
        1
    }
}

/// # Safety
///
/// The returned `Module` owns the heap-allocated `NTT120RefHandle`.
/// `n` must be a power of two >= 2 (asserted by `Module::from_nonnull`).
/// The NTT tables are built for dimension `n`.
unsafe impl ModuleNewImpl<Self> for NTT120Ref {
    fn new_impl(n: u64) -> Module<Self> {
        let handle = NTT120RefHandle {
            table_ntt: NttTable::new(n as usize),
            table_intt: NttTableInv::new(n as usize),
            meta_bbc: BbcMeta::new(),
        };
        let ptr: NonNull<NTT120RefHandle> = NonNull::from(Box::leak(Box::new(handle)));
        unsafe { Module::from_nonnull(ptr, n) }
    }
}

/// # Safety
///
/// The returned references are valid for the lifetime of `&self`.
/// All fields are fully initialised in [`NTT120Ref::new_impl`].
unsafe impl NttHandleProvider for NTT120RefHandle {
    fn get_ntt_table(&self) -> &NttTable<Primes30> {
        &self.table_ntt
    }

    fn get_intt_table(&self) -> &NttTableInv<Primes30> {
        &self.table_intt
    }

    fn get_bbc_meta(&self) -> &BbcMeta<Primes30> {
        &self.meta_bbc
    }
}
