//! Backend handle and module initialisation for [`NTTIfmaRef`](crate::NTTIfmaRef).
//!
//! This module defines:
//!
//! - [`NTTIfmaRefHandle`]: the opaque handle stored inside a `Module<NTTIfmaRef>`,
//!   holding precomputed NTT and iNTT twiddle-factor tables and multiply-accumulate metadata.
//! - The [`Backend`] trait implementation, which defines scalar types and the
//!   handle destruction path.
//! - The [`ModuleNewImpl`] implementation, which allocates the handle on the heap
//!   and transfers ownership to the `Module`.
//! - The [`NttIfmaHandleProvider`] impl for [`NTTIfmaRefHandle`], wiring the handle into
//!   the blanket `NttIfmaModuleHandle` impl provided by `poulpy-hal`.

use std::ptr::NonNull;

use poulpy_hal::{
    layouts::{Backend, Module},
    oep::ModuleNewImpl,
    reference::ntt_ifma::{
        mat_vec::{BbbIfmaMeta, BbcIfmaMeta},
        ntt::{NttIfmaTable, NttIfmaTableInv},
        primes::Primes40,
        vec_znx_dft::NttIfmaHandleProvider,
    },
    reference::ntt120::types::Q120bScalar,
};

use crate::NTTIfmaRef;

/// Opaque handle for the [`NTTIfmaRef`](crate::NTTIfmaRef) backend.
///
/// Holds precomputed twiddle-factor tables for the forward NTT and inverse NTT
/// of size `n`, and the lazy-accumulation metadata for bbc and bbb products
/// using the 3-prime IFMA representation.
///
/// This struct is heap-allocated during module creation and freed when the
/// `Module<NTTIfmaRef>` is dropped (via [`Backend::destroy`]).
#[repr(C)]
pub struct NTTIfmaRefHandle {
    table_ntt: NttIfmaTable<Primes40>,
    table_intt: NttIfmaTableInv<Primes40>,
    meta_bbc: BbcIfmaMeta<Primes40>,
    meta_bbb: BbbIfmaMeta<Primes40>,
}

impl Backend for NTTIfmaRef {
    type ScalarPrep = Q120bScalar;
    type ScalarBig = i128;
    type Handle = NTTIfmaRefHandle;

    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe {
            drop(Box::from_raw(handle.as_ptr()));
        }
    }
}

/// # Safety
///
/// The returned `Module` owns the heap-allocated `NTTIfmaRefHandle`.
/// `n` must be a power of two >= 2 (asserted by `Module::from_nonnull`).
/// The NTT tables are built for dimension `n`.
unsafe impl ModuleNewImpl<Self> for NTTIfmaRef {
    fn new_impl(n: u64) -> Module<Self> {
        let handle = NTTIfmaRefHandle {
            table_ntt: NttIfmaTable::new(n as usize),
            table_intt: NttIfmaTableInv::new(n as usize),
            meta_bbc: BbcIfmaMeta::new(),
            meta_bbb: BbbIfmaMeta::new(),
        };
        let ptr: NonNull<NTTIfmaRefHandle> = NonNull::from(Box::leak(Box::new(handle)));
        unsafe { Module::from_nonnull(ptr, n) }
    }
}

/// # Safety
///
/// The returned references are valid for the lifetime of `&self`.
/// All fields are fully initialised in [`NTTIfmaRef::new_impl`].
unsafe impl NttIfmaHandleProvider for NTTIfmaRefHandle {
    fn get_ntt_ifma_table(&self) -> &NttIfmaTable<Primes40> {
        &self.table_ntt
    }

    fn get_intt_ifma_table(&self) -> &NttIfmaTableInv<Primes40> {
        &self.table_intt
    }

    fn get_bbc_ifma_meta(&self) -> &BbcIfmaMeta<Primes40> {
        &self.meta_bbc
    }

    fn get_bbb_ifma_meta(&self) -> &BbbIfmaMeta<Primes40> {
        &self.meta_bbb
    }
}
