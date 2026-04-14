//! Backend handle and module initialisation for [`NTTIfmaRef`](crate::NTTIfmaRef).
//!
//! This module defines:
//!
//! - [`NTTIfmaRefHandle`]: the opaque handle stored inside a `Module<NTTIfmaRef>`,
//!   holding precomputed NTT and iNTT twiddle-factor tables and multiply-accumulate metadata.
//! - The [`Backend`] trait implementation, which defines scalar types and the
//!   handle destruction path.
//! - The [`NttIfmaHandleFactory`] implementation, which builds the handle stored
//!   inside the `Module`.
//! - The [`NttIfmaHandleProvider`] impl for [`NTTIfmaRefHandle`], wiring the handle into
//!   the blanket `NttIfmaModuleHandle` impl provided by `poulpy-hal`.

use std::ptr::NonNull;

use poulpy_hal::{alloc_aligned, assert_alignment, layouts::Backend};

use crate::reference::ntt_ifma::{
    mat_vec::{BbbIfmaMeta, BbcIfmaMeta},
    ntt::{NttIfmaTable, NttIfmaTableInv},
    primes::Primes40,
    vec_znx_dft::{NttIfmaHandleFactory, NttIfmaHandleProvider},
};
use crate::reference::ntt120::types::Q120bScalar;

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
    type OwnedBuf = Vec<u8>;
    type Handle = NTTIfmaRefHandle;
    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        alloc_aligned::<u8>(len)
    }
    fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf {
        assert_alignment(bytes.as_ptr());
        bytes
    }

    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe {
            drop(Box::from_raw(handle.as_ptr()));
        }
    }
}

/// # Safety
///
/// The returned handle must be fully initialized for `n`.
unsafe impl NttIfmaHandleFactory for NTTIfmaRefHandle {
    fn create_ntt_ifma_handle(n: usize) -> Self {
        NTTIfmaRefHandle {
            table_ntt: NttIfmaTable::new(n),
            table_intt: NttIfmaTableInv::new(n),
            meta_bbc: BbcIfmaMeta::new(),
            meta_bbb: BbbIfmaMeta::new(),
        }
    }
}

/// # Safety
///
/// The returned references are valid for the lifetime of `&self`.
/// All fields are fully initialised in [`NttIfmaHandleFactory::create_ntt_ifma_handle`].
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
