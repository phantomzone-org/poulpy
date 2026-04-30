use anyhow::Result;
use poulpy_core::{GLWEShift, ScratchArenaTakeCore};
use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::layouts::CKKSCiphertext;

/// CKKS rescaling and level-alignment APIs.
///
/// Rescale lowers `log_budget` by shifting the torus representation. Align
/// equalizes the `log_budget` of two ciphertexts by rescaling the one with
/// more remaining capacity.
pub trait CKKSRescaleOps<BE: Backend> {
    /// Returns scratch bytes required by [`Self::ckks_rescale_into`].
    fn ckks_rescale_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    /// Rescales a ciphertext in place by `k` bits.
    ///
    /// Errors include `InsufficientHomomorphicCapacity` if `k` exceeds the
    /// available `log_budget`.
    fn ckks_rescale_assign<D: Data>(
        &self,
        ct: &mut CKKSCiphertext<D>,
        k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        CKKSCiphertext<D>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    /// Computes a rescaled copy of `src` into `dst`.
    ///
    /// Errors include `InsufficientHomomorphicCapacity`.
    fn ckks_rescale_into<Dst: Data, Src: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        k: usize,
        src: &CKKSCiphertext<Src>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    /// Rescales either `a` or `b` in place so both ciphertexts end up with the
    /// same `log_budget`.
    ///
    /// Errors propagate from the underlying rescale operation.
    fn ckks_align_assign<A: Data, B: Data>(
        &self,
        a: &mut CKKSCiphertext<A>,
        b: &mut CKKSCiphertext<B>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<B>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    /// Returns scratch bytes required by [`Self::ckks_align_assign`].
    fn ckks_align_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;
}
