//! Backend handle and module initialisation for [`NTT126Ifma`](super::NTT126Ifma).
//!
//! This module defines:
//!
//! - [`NTT126IfmaHandle`]: the opaque handle stored inside a `Module<NTT126Ifma>`,
//!   holding precomputed NTT and iNTT twiddle-factor tables and multiply-accumulate metadata.
//! - The [`Backend`] trait implementation, which defines scalar types and the
//!   handle destruction path.
//! - The [`NttIfmaHandleFactory`] implementation, which allocates the handle on the heap
//!   and verifies AVX512-IFMA availability at runtime.
//! - The [`NttIfmaHandleProvider`] impl for [`NTT126IfmaHandle`], wiring the handle into
//!   the blanket `NttIfmaModuleHandle` impl provided by `poulpy-cpu-ref`.

use std::ptr::NonNull;

use poulpy_cpu_ref::reference::ntt_ifma::{
    mat_vec::BbcIfmaMeta,
    ntt::{NttIfmaTable, NttIfmaTableInv},
    primes::Primes42,
    vec_znx_dft::{NttIfmaHandleFactory, NttIfmaHandleProvider},
};
use poulpy_cpu_ref::reference::ntt120::types::Q120bScalar;
use poulpy_hal::{alloc_aligned, assert_alignment, layouts::Backend};

use crate::NTT126Ifma;

/// Opaque handle for the [`NTT126Ifma`](super::NTT126Ifma) backend.
///
/// Holds precomputed twiddle-factor tables for the forward NTT and inverse NTT
/// of size `n`, and the lazy-accumulation metadata for `q120b × q120c` products.
///
/// This struct is heap-allocated during module creation and freed when the
/// `Module<NTT126Ifma>` is dropped (via [`Backend::destroy`]).
#[repr(C)]
pub struct NTT126IfmaHandle {
    table_ntt: NttIfmaTable<Primes42>,
    table_intt: NttIfmaTableInv<Primes42>,
    meta_bbc: BbcIfmaMeta<Primes42>,
}

impl Backend for NTT126Ifma {
    type ScalarPrep = Q120bScalar;
    type ScalarBig = i128;
    type OwnedBuf = Vec<u8>;
    type Handle = NTT126IfmaHandle;
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
unsafe impl NttIfmaHandleFactory for NTT126IfmaHandle {
    fn create_ntt_ifma_handle(n: usize) -> Self {
        NTT126IfmaHandle {
            table_ntt: NttIfmaTable::new(n),
            table_intt: NttIfmaTableInv::new(n),
            meta_bbc: BbcIfmaMeta::new(),
        }
    }

    fn assert_ntt_ifma_runtime_support() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::arch::is_x86_feature_detected!("avx512f") {
                panic!("NTT126Ifma requires x86_64 with AVX512-F support");
            }
            if !std::arch::is_x86_feature_detected!("avx512ifma") {
                panic!("NTT126Ifma requires x86_64 with AVX512-IFMA support");
            }
            if !std::arch::is_x86_feature_detected!("avx512vl") {
                panic!("NTT126Ifma requires x86_64 with AVX512-VL support");
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        panic!("NTT126Ifma requires x86_64 with AVX512-F + AVX512-IFMA + AVX512-VL support");
    }
}

/// # Safety
///
/// The returned references are valid for the lifetime of `&self`.
/// All fields are fully initialised in [`NttIfmaHandleFactory::create_ntt_ifma_handle`].
unsafe impl NttIfmaHandleProvider for NTT126IfmaHandle {
    fn get_ntt_ifma_table(&self) -> &NttIfmaTable<Primes42> {
        &self.table_ntt
    }

    fn get_intt_ifma_table(&self) -> &NttIfmaTableInv<Primes42> {
        &self.table_intt
    }

    fn get_bbc_ifma_meta(&self) -> &BbcIfmaMeta<Primes42> {
        &self.meta_bbc
    }
}
