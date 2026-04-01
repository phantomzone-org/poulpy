//! CKKS tensor product result (intermediate between tensor and relinearize).

use poulpy_core::layouts::{Base2K, Degree, GLWEInfos, GLWETensor, Rank, TorusPrecision};
use poulpy_hal::layouts::Data;

pub struct CKKSTensor<D: Data> {
    pub inner: GLWETensor<D>,
    pub log_delta: u32,
}

impl CKKSTensor<Vec<u8>> {
    /// Allocates a tensor for the product of two rank-1 CKKS ciphertexts.
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, log_delta: u32) -> Self {
        CKKSTensor {
            inner: GLWETensor::alloc(n, base2k, k, Rank(1)),
            log_delta,
        }
    }

    /// Allocates a tensor from a GLWE layout, setting `log_delta`.
    pub fn alloc_from_infos<A: GLWEInfos>(infos: &A, log_delta: u32) -> Self {
        CKKSTensor {
            inner: GLWETensor::alloc_from_infos(infos),
            log_delta,
        }
    }
}
