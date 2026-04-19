use crate::{
    api::{
        VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAccumulate, VmpApplyDftToDftAccumulateTmpBytes,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPMatBytesOf, VmpPrepare, VmpPrepareTmpBytes, VmpZero,
    },
    layouts::{
        Backend, MatZnxToRef, Module, Scratch, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, VmpPMatOwned, VmpPMatToMut,
        VmpPMatToRef,
    },
    oep::HalImpl,
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
    B: Backend + HalImpl<B>,
{
    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        <B as HalImpl<B>>::vmp_prepare_tmp_bytes(self, rows, cols_in, cols_out, size)
    }
}

impl<B> VmpPrepare<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vmp_prepare<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<B>)
    where
        R: VmpPMatToMut<B>,
        A: MatZnxToRef,
    {
        <B as HalImpl<B>>::vmp_prepare(self, res, a, scratch)
    }
}

impl<B> VmpApplyDftTmpBytes for Module<B>
where
    B: Backend + HalImpl<B>,
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
        <B as HalImpl<B>>::vmp_apply_dft_tmp_bytes(self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
    }
}

impl<B> VmpApplyDft<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vmp_apply_dft<R, A, C>(&self, res: &mut R, a: &A, b: &C, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef,
        C: VmpPMatToRef<B>,
    {
        <B as HalImpl<B>>::vmp_apply_dft(self, res, a, b, scratch);
    }
}

impl<B> VmpApplyDftToDftTmpBytes for Module<B>
where
    B: Backend + HalImpl<B>,
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
        <B as HalImpl<B>>::vmp_apply_dft_to_dft_tmp_bytes(self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
    }
}

impl<B> VmpApplyDftToDft<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vmp_apply_dft_to_dft<R, A, C>(&self, res: &mut R, a: &A, b: &C, limb_offset: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>,
    {
        <B as HalImpl<B>>::vmp_apply_dft_to_dft(self, res, a, b, limb_offset, scratch);
    }
}

impl<B> VmpApplyDftToDftAccumulateTmpBytes for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vmp_apply_dft_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        <B as HalImpl<B>>::vmp_apply_dft_to_dft_accumulate_tmp_bytes(
            self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
        )
    }
}

impl<B> VmpApplyDftToDftAccumulate<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vmp_apply_dft_to_dft_accumulate<R, A, C>(&self, res: &mut R, a: &A, b: &C, limb_offset: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>,
    {
        <B as HalImpl<B>>::vmp_apply_dft_to_dft_accumulate(self, res, a, b, limb_offset, scratch);
    }
}

impl<B> VmpZero<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vmp_zero<R>(&self, res: &mut R)
    where
        R: VmpPMatToMut<B>,
    {
        <B as HalImpl<B>>::vmp_zero(self, res);
    }
}
