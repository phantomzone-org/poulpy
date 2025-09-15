use crate::{
    api::{
        VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftAddTmpBytes,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatFromBytes, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{
        Backend, MatZnxToRef, Module, Scratch, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, VmpPMatOwned, VmpPMatToMut,
        VmpPMatToRef,
    },
    oep::{
        VmpApplyDftImpl, VmpApplyDftTmpBytesImpl, VmpApplyDftToDftAddImpl, VmpApplyDftToDftAddTmpBytesImpl, VmpApplyDftToDftImpl,
        VmpApplyDftToDftTmpBytesImpl, VmpPMatAllocBytesImpl, VmpPMatAllocImpl, VmpPMatFromBytesImpl, VmpPrepareImpl,
        VmpPrepareTmpBytesImpl,
    },
};

impl<B> VmpPMatAlloc<B> for Module<B>
where
    B: Backend + VmpPMatAllocImpl<B>,
{
    fn vmp_pmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<B> {
        B::vmp_pmat_alloc_impl(self.n(), rows, cols_in, cols_out, size)
    }
}

impl<B> VmpPMatAllocBytes for Module<B>
where
    B: Backend + VmpPMatAllocBytesImpl<B>,
{
    fn vmp_pmat_alloc_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        B::vmp_pmat_alloc_bytes_impl(self.n(), rows, cols_in, cols_out, size)
    }
}

impl<B> VmpPMatFromBytes<B> for Module<B>
where
    B: Backend + VmpPMatFromBytesImpl<B>,
{
    fn vmp_pmat_from_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize, bytes: Vec<u8>) -> VmpPMatOwned<B> {
        B::vmp_pmat_from_bytes_impl(self.n(), rows, cols_in, cols_out, size, bytes)
    }
}

impl<B> VmpPrepareTmpBytes for Module<B>
where
    B: Backend + VmpPrepareTmpBytesImpl<B>,
{
    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        B::vmp_prepare_tmp_bytes_impl(self, rows, cols_in, cols_out, size)
    }
}

impl<B> VmpPrepare<B> for Module<B>
where
    B: Backend + VmpPrepareImpl<B>,
{
    fn vmp_prepare<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<B>)
    where
        R: VmpPMatToMut<B>,
        A: MatZnxToRef,
    {
        B::vmp_prepare_impl(self, res, a, scratch)
    }
}

impl<B> VmpApplyDftTmpBytes for Module<B>
where
    B: Backend + VmpApplyDftTmpBytesImpl<B>,
{
    fn vmp_apply_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_dft_tmp_bytes_impl(
            self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
        )
    }
}

impl<B> VmpApplyDft<B> for Module<B>
where
    B: Backend + VmpApplyDftImpl<B>,
{
    fn vmp_apply_dft<R, A, C>(&self, res: &mut R, a: &A, b: &C, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef,
        C: VmpPMatToRef<B>,
    {
        B::vmp_apply_dft_impl(self, res, a, b, scratch);
    }
}

impl<B> VmpApplyDftToDftTmpBytes for Module<B>
where
    B: Backend + VmpApplyDftToDftTmpBytesImpl<B>,
{
    fn vmp_apply_dft_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_dft_to_dft_tmp_bytes_impl(
            self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
        )
    }
}

impl<B> VmpApplyDftToDft<B> for Module<B>
where
    B: Backend + VmpApplyDftToDftImpl<B>,
{
    fn vmp_apply_dft_to_dft<R, A, C>(&self, res: &mut R, a: &A, b: &C, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>,
    {
        B::vmp_apply_dft_to_dft_impl(self, res, a, b, scratch);
    }
}

impl<B> VmpApplyDftToDftAddTmpBytes for Module<B>
where
    B: Backend + VmpApplyDftToDftAddTmpBytesImpl<B>,
{
    fn vmp_apply_dft_to_dft_add_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_dft_to_dft_add_tmp_bytes_impl(
            self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
        )
    }
}

impl<B> VmpApplyDftToDftAdd<B> for Module<B>
where
    B: Backend + VmpApplyDftToDftAddImpl<B>,
{
    fn vmp_apply_dft_to_dft_add<R, A, C>(&self, res: &mut R, a: &A, b: &C, scale: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>,
    {
        B::vmp_apply_dft_to_dft_add_impl(self, res, a, b, scale, scratch);
    }
}
