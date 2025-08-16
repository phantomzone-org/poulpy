use crate::hal::{
    api::{
        VmpApply, VmpApplyAdd, VmpApplyAddTmpBytes, VmpApplyTmpBytes, VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatFromBytes,
        VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{Backend, MatZnxToRef, Module, Scratch, VecZnxDftToMut, VecZnxDftToRef, VmpPMatOwned, VmpPMatToMut, VmpPMatToRef},
    oep::{
        VmpApplyAddImpl, VmpApplyAddTmpBytesImpl, VmpApplyImpl, VmpApplyTmpBytesImpl, VmpPMatAllocBytesImpl, VmpPMatAllocImpl,
        VmpPMatFromBytesImpl, VmpPMatPrepareImpl, VmpPrepareTmpBytesImpl,
    },
};

impl<B> VmpPMatAlloc<B> for Module<B>
where
    B: Backend + VmpPMatAllocImpl<B>,
{
    fn vmp_pmat_alloc(&self, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<B> {
        B::vmp_pmat_alloc_impl(n, rows, cols_in, cols_out, size)
    }
}

impl<B> VmpPMatAllocBytes for Module<B>
where
    B: Backend + VmpPMatAllocBytesImpl<B>,
{
    fn vmp_pmat_alloc_bytes(&self, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        B::vmp_pmat_alloc_bytes_impl(n, rows, cols_in, cols_out, size)
    }
}

impl<B> VmpPMatFromBytes<B> for Module<B>
where
    B: Backend + VmpPMatFromBytesImpl<B>,
{
    fn vmp_pmat_from_bytes(
        &self,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> VmpPMatOwned<B> {
        B::vmp_pmat_from_bytes_impl(n, rows, cols_in, cols_out, size, bytes)
    }
}

impl<B> VmpPrepareTmpBytes for Module<B>
where
    B: Backend + VmpPrepareTmpBytesImpl<B>,
{
    fn vmp_prepare_tmp_bytes(&self, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        B::vmp_prepare_tmp_bytes_impl(self, n, rows, cols_in, cols_out, size)
    }
}

impl<B> VmpPrepare<B> for Module<B>
where
    B: Backend + VmpPMatPrepareImpl<B>,
{
    fn vmp_prepare<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<B>)
    where
        R: VmpPMatToMut<B>,
        A: MatZnxToRef,
    {
        B::vmp_prepare_impl(self, res, a, scratch)
    }
}

impl<B> VmpApplyTmpBytes for Module<B>
where
    B: Backend + VmpApplyTmpBytesImpl<B>,
{
    fn vmp_apply_tmp_bytes(
        &self,
        n: usize,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_tmp_bytes_impl(
            self, n, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
        )
    }
}

impl<B> VmpApply<B> for Module<B>
where
    B: Backend + VmpApplyImpl<B>,
{
    fn vmp_apply<R, A, C>(&self, res: &mut R, a: &A, b: &C, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>,
    {
        B::vmp_apply_impl(self, res, a, b, scratch);
    }
}

impl<B> VmpApplyAddTmpBytes for Module<B>
where
    B: Backend + VmpApplyAddTmpBytesImpl<B>,
{
    fn vmp_apply_add_tmp_bytes(
        &self,
        n: usize,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_add_tmp_bytes_impl(
            self, n, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
        )
    }
}

impl<B> VmpApplyAdd<B> for Module<B>
where
    B: Backend + VmpApplyAddImpl<B>,
{
    fn vmp_apply_add<R, A, C>(&self, res: &mut R, a: &A, b: &C, scale: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>,
    {
        B::vmp_apply_add_impl(self, res, a, b, scale, scratch);
    }
}
