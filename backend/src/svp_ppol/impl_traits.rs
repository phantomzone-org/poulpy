use crate::{Backend, Module, ScalarZnxToRef, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef};

pub unsafe trait SvpPPolFromBytesImpl<B: Backend> {
    fn svp_ppol_from_bytes_impl(module: &Module<B>, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<B>;
}

pub unsafe trait SvpPPolAllocImpl<B: Backend> {
    fn svp_ppol_alloc_impl(module: &Module<B>, cols: usize) -> SvpPPolOwned<B>;
}

pub unsafe trait SvpPPolAllocBytesImpl<B: Backend> {
    fn svp_ppol_alloc_bytes_impl(module: &Module<B>, cols: usize) -> usize;
}

pub unsafe trait SvpPrepareImpl<B: Backend> {
    fn svp_prepare_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<B>,
        A: ScalarZnxToRef;
}

pub unsafe trait SvpApplyImpl<B: Backend> {
    fn svp_apply_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxDftToRef<B>;
}

// -------------------- OPEN EXTENSION POINT (unsafe) --------------------
/// # Safety
/// Implementors must uphold all of the following for **every** call:
///
/// * **Memory domains**: Pointers produced by `to_ref()` / `to_mut()` must be valid
///   in the target execution domain for `Self` (e.g., CPU host memory for CPU,
///   device memory for a specific GPU). If host↔device transfers are required,
///   perform them inside the implementation; do not assume the caller synchronized.
///
/// * **Alignment & layout**: All data must match the layout, stride, and element
///   size expected by the kernel. `size()`, `rows()`, `cols_in()`, `cols_out()`
///   and `n()` must be interpreted identically to the reference CPU implementation.
///
/// * **Scratch lifetime**: Any scratch obtained from `scratch.tmp_slice(...)` (or a
///   backend-specific variant) must remain valid for the duration of the call; it
///   may be reused by the caller afterwards. Do not retain pointers past return.
///
/// * **Synchronization**: The call must appear **logically synchronous** to the
///   caller. If you enqueue asynchronous work (e.g., CUDA streams), you must
///   ensure completion before returning or clearly document and implement a
///   synchronization contract used by all backends consistently.
///
/// * **Aliasing & overlaps**: If `res`, `a`, `b` alias or overlap in ways that
///   violate your kernel’s requirements, you must either handle safely or reject
///   with a defined error path (e.g., debug assert). Never trigger UB.
///
/// * **Numerical contract**: For modular/integer transforms, results must be
///   bit-exact to the specification. For floating-point, any permitted tolerance
///   must be documented and consistent with the crate’s guarantees.
pub unsafe trait SvpApplyInplaceImpl: Backend {
    fn svp_apply_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: SvpPPolToRef<Self>;
}
