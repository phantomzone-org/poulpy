//! Backend handle and module initialisation for [`NTT120Avx`](super::NTT120Avx).
//!
//! This module defines:
//!
//! - [`NTT120AvxHandle`]: the opaque handle stored inside a `Module<NTT120Avx>`,
//!   holding precomputed NTT and iNTT twiddle-factor tables and multiply-accumulate metadata.
//! - The [`Backend`] trait implementation, which defines scalar types and the
//!   handle destruction path.
//! - The [`NttHandleFactory`] implementation, which allocates the handle on the heap
//!   and verifies AVX2 availability at runtime.
//! - The [`NttHandleProvider`] impl for [`NTT120AvxHandle`], wiring the handle into
//!   the blanket `NttModuleHandle` impl provided by `poulpy-hal`.

use std::ptr::NonNull;

use poulpy_cpu_ref::reference::ntt120::{
    mat_vec::{BbbMeta, BbcMeta},
    ntt::{NttTable, NttTableInv},
    primes::Primes30,
    types::Q120bScalar,
    vec_znx_dft::{NttHandleFactory, NttHandleProvider},
};
use poulpy_hal::{alloc_aligned, assert_alignment, layouts::Backend};

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
    type OwnedBuf = Vec<u8>;
    type Handle = NTT120AvxHandle;
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
///
/// # Panics
///
/// Panics if the runtime CPU does not support the AVX2 instruction set.
unsafe impl NttHandleFactory for NTT120AvxHandle {
    fn create_ntt_handle(n: usize) -> Self {
        NTT120AvxHandle {
            table_ntt: NttTable::new(n),
            table_intt: NttTableInv::new(n),
            meta_bbc: BbcMeta::new(),
            meta_bbb: BbbMeta::new(),
        }
    }

    fn assert_ntt_runtime_support() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            panic!("arch must support avx2")
        }
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
