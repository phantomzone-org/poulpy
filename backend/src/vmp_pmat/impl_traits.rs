use crate::{Backend, MatZnxToRef, Module, Scratch, VecZnxDftToMut, VecZnxDftToRef, VmpPMatOwned, VmpPMatToMut, VmpPMatToRef};

pub trait VmpPMatAllocImpl<B: Backend> {
    fn vmp_pmat_alloc_impl(module: &Module<B>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<B>;
}

pub trait VmpPMatAllocBytesImpl<B: Backend> {
    fn vmp_pmat_alloc_bytes_impl(module: &Module<B>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

pub trait VmpPMatFromBytesImpl<B: Backend> {
    fn vmp_pmat_from_bytes_impl(
        module: &Module<B>,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> VmpPMatOwned<B>;
}

pub trait VmpPrepareTmpBytesImpl<B: Backend> {
    fn vmp_prepare_tmp_bytes_impl(module: &Module<B>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

pub trait VmpPMatPrepareImpl<B: Backend> {
    fn vmp_prepare_impl<R, A>(module: &Module<B>, res: &mut R, a: &A, scratch: &mut Scratch)
    where
        R: VmpPMatToMut<B>,
        A: MatZnxToRef;
}

pub trait VmpApplyTmpBytesImpl<B: Backend> {
    fn vmp_apply_tmp_bytes_impl(
        module: &Module<B>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

pub trait VmpApplyImpl<B: Backend> {
    fn vmp_apply_impl<R, A, C>(module: &Module<B>, res: &mut R, a: &A, b: &C, scratch: &mut Scratch)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>;
}

pub trait VmpApplyAddTmpBytesImpl<B: Backend> {
    fn vmp_apply_add_tmp_bytes_impl(
        module: &Module<B>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

pub trait VmpApplyAddImpl<B: Backend> {
    // Same as [MatZnxDftOps::vmp_apply] except result is added on R instead of overwritting R.
    fn vmp_apply_add_impl<R, A, C>(module: &Module<B>, res: &mut R, a: &A, b: &C, scale: usize, scratch: &mut Scratch)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>;
}
