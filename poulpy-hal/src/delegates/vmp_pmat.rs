use crate::{
    api::{
        VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPMatBytesOf, VmpPrepare,
        VmpPrepareTmpBytes, VmpZero,
    },
    layouts::{
        Backend, MatZnxToRef, Module, Scratch, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, VmpPMatOwned, VmpPMatToMut,
        VmpPMatToRef,
    },
    oep::{
        VmpApplyDftImpl, VmpApplyDftTmpBytesImpl, VmpApplyDftToDftImpl, VmpApplyDftToDftTmpBytesImpl, VmpPrepareImpl,
        VmpPrepareTmpBytesImpl, VmpZeroImpl,
    },
};

impl<B: Backend> VmpPMatAlloc<B> for Module<B> {
    fn vmp_pmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<B> {
        VmpPMatOwned::alloc(self.n(), rows, cols_in, cols_out, size)
    }
}

impl<B: Backend> VmpPMatBytesOf for Module<B> {
    fn bytes_of_vmp_pmat(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        B::bytes_of_vmp_pmat(self.n(), rows, cols_in, cols_out, size)
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
        B::vmp_apply_dft_tmp_bytes_impl(self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
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
        B::vmp_apply_dft_to_dft_tmp_bytes_impl(self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
    }
}

impl<B> VmpApplyDftToDft<B> for Module<B>
where
    B: Backend + VmpApplyDftToDftImpl<B>,
{
    fn vmp_apply_dft_to_dft<R, A, C>(&self, res: &mut R, a: &A, b: &C, limb_offset: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>,
    {
        B::vmp_apply_dft_to_dft_impl(self, res, a, b, limb_offset, scratch);
    }
}

impl<B> VmpZero<B> for Module<B>
where
    B: Backend + VmpZeroImpl<B>,
{
    fn vmp_zero<R>(&self, res: &mut R)
    where
        R: VmpPMatToMut<B>,
    {
        B::vmp_zero_impl(self, res);
    }
}
