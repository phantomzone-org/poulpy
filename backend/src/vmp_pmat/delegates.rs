use crate::{
    Backend, MatZnxToRef, Module, Scratch, VecZnxDftToMut, VecZnxDftToRef, VmpApply, VmpApplyAdd, VmpApplyAddTmpBytes,
    VmpApplyTmpBytes, VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatFromBytes, VmpPMatOwned, VmpPMatPrepare, VmpPMatToMut,
    VmpPMatToRef, VmpPrepareTmpBytes,
    vmp_pmat::impl_traits::{
        VmpApplyAddImpl, VmpApplyAddTmpBytesImpl, VmpApplyImpl, VmpApplyTmpBytesImpl, VmpPMatAllocBytesImpl, VmpPMatAllocImpl,
        VmpPMatFromBytesImpl, VmpPMatPrepareImpl, VmpPrepareTmpBytesImpl,
    },
};

impl<B: Backend> VmpPMatAlloc<B> for Module<B>
where
    (): VmpPMatAllocImpl<B>,
{
    fn vmp_pmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<B> {
        <() as VmpPMatAllocImpl<B>>::vmp_pmat_alloc_impl(self, rows, cols_in, cols_out, size)
    }
}

impl<B: Backend> VmpPMatAllocBytes for Module<B>
where
    (): VmpPMatAllocBytesImpl<B>,
{
    fn vmp_pmat_alloc_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        <() as VmpPMatAllocBytesImpl<B>>::vmp_pmat_alloc_bytes_impl(self, rows, cols_in, cols_out, size)
    }
}

impl<B: Backend> VmpPMatFromBytes<B> for Module<B>
where
    (): VmpPMatFromBytesImpl<B>,
{
    fn vmp_pmat_from_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize, bytes: Vec<u8>) -> VmpPMatOwned<B> {
        <() as VmpPMatFromBytesImpl<B>>::vmp_pmat_from_bytes_impl(self, rows, cols_in, cols_out, size, bytes)
    }
}

impl<B: Backend> VmpPrepareTmpBytes for Module<B>
where
    (): VmpPrepareTmpBytesImpl<B>,
{
    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        <() as VmpPrepareTmpBytesImpl<B>>::vmp_prepare_tmp_bytes_impl(self, rows, cols_in, cols_out, size)
    }
}

impl<B: Backend> VmpPMatPrepare<B> for Module<B>
where
    (): VmpPMatPrepareImpl<B>,
{
    fn vmp_prepare<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch)
    where
        R: VmpPMatToMut<B>,
        A: MatZnxToRef,
    {
        <() as VmpPMatPrepareImpl<B>>::vmp_prepare_impl(self, res, a, scratch)
    }
}

impl<B: Backend> VmpApplyTmpBytes for Module<B>
where
    (): VmpApplyTmpBytesImpl<B>,
{
    fn vmp_apply_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        <() as VmpApplyTmpBytesImpl<B>>::vmp_apply_tmp_bytes_impl(
            self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
        )
    }
}

impl<B: Backend> VmpApply<B> for Module<B>
where
    (): VmpApplyImpl<B>,
{
    fn vmp_apply<R, A, C>(&self, res: &mut R, a: &A, b: &C, scratch: &mut Scratch)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>,
    {
        <() as VmpApplyImpl<B>>::vmp_apply_impl(self, res, a, b, scratch);
    }
}

impl<B: Backend> VmpApplyAddTmpBytes for Module<B>
where
    (): VmpApplyAddTmpBytesImpl<B>,
{
    fn vmp_apply_add_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        <() as VmpApplyAddTmpBytesImpl<B>>::vmp_apply_add_tmp_bytes_impl(
            self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
        )
    }
}

impl<B: Backend> VmpApplyAdd<B> for Module<B>
where
    (): VmpApplyAddImpl<B>,
{
    fn vmp_apply_add<R, A, C>(&self, res: &mut R, a: &A, b: &C, scale: usize, scratch: &mut Scratch)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>,
    {
        <() as VmpApplyAddImpl<B>>::vmp_apply_add_impl(self, res, a, b, scale, scratch);
    }
}
