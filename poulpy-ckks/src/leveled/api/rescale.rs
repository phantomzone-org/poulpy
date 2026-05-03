use anyhow::Result;
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSInfos, SetCKKSInfos};

/// CKKS rescaling and level-alignment APIs.
///
/// Rescale lowers `log_budget` by shifting the torus representation. Align
/// equalizes the `log_budget` of two ciphertexts by rescaling the one with
/// more remaining capacity.
pub trait CKKSRescaleOps<BE: Backend> {
    /// Returns scratch bytes required by [`Self::ckks_rescale_into`].
    fn ckks_rescale_tmp_bytes(&self) -> usize;

    /// Rescales a ciphertext in place by `k` bits.
    ///
    /// Errors include `InsufficientHomomorphicCapacity` if `k` exceeds the
    /// available `log_budget`.
    fn ckks_rescale_assign<Dst>(&self, ct: &mut Dst, k: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos;

    /// Computes a rescaled copy of `src` into `dst`.
    ///
    /// Errors include `InsufficientHomomorphicCapacity`.
    fn ckks_rescale_into<Dst, Src>(&self, dst: &mut Dst, k: usize, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos;

    /// Rescales either `a` or `b` in place so both ciphertexts end up with the
    /// same `log_budget`.
    ///
    /// Errors propagate from the underlying rescale operation.
    fn ckks_align_assign<A, B>(&self, a: &mut A, b: &mut B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        A: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        B: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos;

    /// Returns scratch bytes required by [`Self::ckks_align_assign`].
    fn ckks_align_tmp_bytes(&self) -> usize;
}
