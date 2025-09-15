use crate::layouts::{
    Backend, Data, Module, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftOwned, VecZnxDftToMut, VecZnxDftToRef,
    VecZnxToRef,
};

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDftAllocImpl<B: Backend> {
    fn vec_znx_dft_alloc_impl(n: usize, cols: usize, size: usize) -> VecZnxDftOwned<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDftFromBytesImpl<B: Backend> {
    fn vec_znx_dft_from_bytes_impl(n: usize, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDftApplyImpl<B: Backend> {
    fn vec_znx_dft_apply_impl<R, A>(
        module: &Module<B>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDftAllocBytesImpl<B: Backend> {
    fn vec_znx_dft_alloc_bytes_impl(n: usize, cols: usize, size: usize) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxIdftApplyTmpBytesImpl<B: Backend> {
    fn vec_znx_idft_apply_tmp_bytes_impl(module: &Module<B>) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxIdftApplyImpl<B: Backend> {
    fn vec_znx_idft_apply_impl<R, A>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToRef<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxIdftApplyTmpAImpl<B: Backend> {
    fn vec_znx_idft_apply_tmpa_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToMut<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxIdftApplyConsumeImpl<B: Backend> {
    fn vec_znx_idft_apply_consume_impl<D: Data>(module: &Module<B>, a: VecZnxDft<D, B>) -> VecZnxBig<D, B>
    where
        VecZnxDft<D, B>: VecZnxDftToMut<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDftAddImpl<B: Backend> {
    fn vec_znx_dft_add_impl<R, A, D>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDftAddInplaceImpl<B: Backend> {
    fn vec_znx_dft_add_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDftSubImpl<B: Backend> {
    fn vec_znx_dft_sub_impl<R, A, D>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDftSubABInplaceImpl<B: Backend> {
    fn vec_znx_dft_sub_ab_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDftSubBAInplaceImpl<B: Backend> {
    fn vec_znx_dft_sub_ba_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDftCopyImpl<B: Backend> {
    fn vec_znx_dft_copy_impl<R, A>(
        module: &Module<B>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See TODO for reference code.
/// * See TODO for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait VecZnxDftZeroImpl<B: Backend> {
    fn vec_znx_dft_zero_impl<R>(module: &Module<B>, res: &mut R)
    where
        R: VecZnxDftToMut<B>;
}
