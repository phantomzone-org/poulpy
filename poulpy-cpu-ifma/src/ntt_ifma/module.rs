//! Backend handle and module initialisation for [`NTTIfma`](super::NTTIfma).
//!
//! This module defines:
//!
//! - [`NTTIfmaHandle`]: the opaque handle stored inside a `Module<NTTIfma>`,
//!   holding precomputed NTT and iNTT twiddle-factor tables and multiply-accumulate metadata.
//! - The [`Backend`] trait implementation, which defines scalar types and the
//!   handle destruction path.
//! - The [`NttIfmaHandleFactory`] implementation, which allocates the handle on the heap
//!   and verifies AVX512-IFMA availability at runtime.
//! - The [`NttIfmaHandleProvider`] impl for [`NTTIfmaHandle`], wiring the handle into
//!   the blanket `NttIfmaModuleHandle` impl provided by `poulpy-cpu-ref`.

use std::ptr::NonNull;

use poulpy_cpu_ref::reference::ntt_ifma::{
    mat_vec::{BbbIfmaMeta, BbcIfmaMeta},
    ntt::{NttIfmaTable, NttIfmaTableInv},
    primes::Primes40,
    vec_znx_dft::{NttIfmaHandleFactory, NttIfmaHandleProvider},
};
use poulpy_cpu_ref::reference::ntt120::types::Q120bScalar;
use poulpy_hal::{alloc_aligned, assert_alignment, layouts::Backend};

use crate::NTTIfma;

/// Opaque handle for the [`NTTIfma`](super::NTTIfma) backend.
///
/// Holds precomputed twiddle-factor tables for the forward NTT and inverse NTT
/// of size `n`, and the lazy-accumulation metadata for `q120b × q120c` and
/// `q120b × q120b` products.
///
/// This struct is heap-allocated during module creation and freed when the
/// `Module<NTTIfma>` is dropped (via [`Backend::destroy`]).
#[repr(C)]
pub struct NTTIfmaHandle {
    table_ntt: NttIfmaTable<Primes40>,
    table_intt: NttIfmaTableInv<Primes40>,
    meta_bbc: BbcIfmaMeta<Primes40>,
    meta_bbb: BbbIfmaMeta<Primes40>,
}

impl Backend for NTTIfma {
    type ScalarPrep = Q120bScalar;
    type ScalarBig = i128;
    type OwnedBuf = Vec<u8>;
    type Handle = NTTIfmaHandle;
    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        alloc_aligned::<u8>(len)
    }
    fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf {
        assert_alignment(bytes.as_ptr());
        bytes
    }

    fn bytes_of_vmp_pmat(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        // Prime-major layout: 3 planes (one per CRT prime), no padding lane.
        // Per coefficient: 3 × u64 instead of the default 4 × u64.
        n * rows * cols_in * cols_out * size * 3 * size_of::<u64>()
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
/// Panics if the runtime CPU does not support the AVX512-IFMA instruction set.
unsafe impl NttIfmaHandleFactory for NTTIfmaHandle {
    fn create_ntt_ifma_handle(n: usize) -> Self {
        NTTIfmaHandle {
            table_ntt: NttIfmaTable::new(n),
            table_intt: NttIfmaTableInv::new(n),
            meta_bbc: BbcIfmaMeta::new(),
            meta_bbb: BbbIfmaMeta::new(),
        }
    }

    fn assert_ntt_ifma_runtime_support() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::arch::is_x86_feature_detected!("avx512ifma") {
                panic!("NTTIfma requires x86_64 with AVX512-IFMA support");
            }
            if !std::arch::is_x86_feature_detected!("avx512vl") {
                panic!("NTTIfma requires x86_64 with AVX512-VL support");
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        panic!("NTTIfma requires x86_64 with AVX512-IFMA + AVX512-VL support");
    }
}

/// # Safety
///
/// The returned references are valid for the lifetime of `&self`.
/// All fields are fully initialised in [`NttIfmaHandleFactory::create_ntt_ifma_handle`].
unsafe impl NttIfmaHandleProvider for NTTIfmaHandle {
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
