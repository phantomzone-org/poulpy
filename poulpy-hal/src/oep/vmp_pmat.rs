use crate::layouts::{
    Backend, MatZnxToRef, Module, Scratch, VecZnxDftToMut, VecZnxDftToRef, VmpPMatOwned, VmpPMatToMut, VmpPMatToRef,
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
    fn vmp_prepare_tmp_bytes_impl(
        module: &Module<B>,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VmpPMatPrepareImpl<B: Backend> {
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
pub unsafe trait VmpApplyTmpBytesImpl<B: Backend> {
    fn vmp_apply_tmp_bytes_impl(
        module: &Module<B>,
        n: usize,
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
pub unsafe trait VmpApplyImpl<B: Backend> {
    fn vmp_apply_impl<R, A, C>(module: &Module<B>, res: &mut R, a: &A, b: &C, scratch: &mut Scratch<B>)
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
pub unsafe trait VmpApplyAddTmpBytesImpl<B: Backend> {
    fn vmp_apply_add_tmp_bytes_impl(
        module: &Module<B>,
        n: usize,
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
pub unsafe trait VmpApplyAddImpl<B: Backend> {
    // Same as [MatZnxDftOps::vmp_apply] except result is added on R instead of overwritting R.
    fn vmp_apply_add_impl<R, A, C>(module: &Module<B>, res: &mut R, a: &A, b: &C, scale: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>;
}
