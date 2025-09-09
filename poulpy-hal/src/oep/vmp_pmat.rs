use crate::layouts::{
    Backend, MatZnxToRef, Module, Scratch, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, VmpPMatOwned, VmpPMatToMut, VmpPMatToRef,
};

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpPMatAllocImpl<B: Backend> {
    fn vmp_pmat_alloc_impl(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpPMatAllocBytesImpl<B: Backend> {
    fn vmp_pmat_alloc_bytes_impl(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpPMatFromBytesImpl<B: Backend> {
    fn vmp_pmat_from_bytes_impl(
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> VmpPMatOwned<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpPrepareTmpBytesImpl<B: Backend> {
    fn vmp_prepare_tmp_bytes_impl(module: &Module<B>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpPrepareImpl<B: Backend> {
    fn vmp_prepare_impl<R, A>(module: &Module<B>, res: &mut R, a: &A, scratch: &mut Scratch<B>)
    where
        R: VmpPMatToMut<B>,
        A: MatZnxToRef;
}

#[allow(clippy::too_many_arguments)]
/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpApplyDftTmpBytesImpl<B: Backend> {
    fn vmp_apply_dft_tmp_bytes_impl(
        module: &Module<B>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpApplyDftImpl<B: Backend> {
    fn vmp_apply_dft_impl<R, A, C>(module: &Module<B>, res: &mut R, a: &A, b: &C, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef,
        C: VmpPMatToRef<B>;
}

#[allow(clippy::too_many_arguments)]
/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpApplyDftToDftTmpBytesImpl<B: Backend> {
    fn vmp_apply_dft_to_dft_tmp_bytes_impl(
        module: &Module<B>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpApplyDftToDftImpl<B: Backend> {
    fn vmp_apply_dft_to_dft_impl<R, A, C>(module: &Module<B>, res: &mut R, a: &A, b: &C, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>;
}

#[allow(clippy::too_many_arguments)]
/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpApplyDftToDftAddTmpBytesImpl<B: Backend> {
    fn vmp_apply_dft_to_dft_add_tmp_bytes_impl(
        module: &Module<B>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpApplyDftToDftAddImpl<B: Backend> {
    // Same as [MatZnxDftOps::vmp_apply] except result is added on R instead of overwritting R.
    fn vmp_apply_dft_to_dft_add_impl<R, A, C>(
        module: &Module<B>,
        res: &mut R,
        a: &A,
        b: &C,
        scale: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>;
}
