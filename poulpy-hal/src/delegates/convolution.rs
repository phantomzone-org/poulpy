use crate::{
    api::{CnvPVecAlloc, CnvPVecBytesOf, Convolution},
    layouts::{
        Backend, CnvPVecL, CnvPVecLToMut, CnvPVecLToRef, CnvPVecR, CnvPVecRToMut, CnvPVecRToRef, DeviceBuf, Module, Scratch,
        VecZnxBigToMut, VecZnxDftToMut, VecZnxToRef, ZnxInfos, ZnxViewMut,
    },
    oep::HalImpl,
};

impl<BE: Backend> CnvPVecAlloc<BE> for Module<BE> {
    fn cnv_pvec_left_alloc(&self, cols: usize, size: usize) -> CnvPVecL<DeviceBuf<BE>, BE> {
        CnvPVecL::alloc(self.n(), cols, size)
    }

    fn cnv_pvec_right_alloc(&self, cols: usize, size: usize) -> CnvPVecR<DeviceBuf<BE>, BE> {
        CnvPVecR::alloc(self.n(), cols, size)
    }
}

impl<BE: Backend> CnvPVecBytesOf for Module<BE> {
    fn bytes_of_cnv_pvec_left(&self, cols: usize, size: usize) -> usize {
        BE::bytes_of_cnv_pvec_left(self.n(), cols, size)
    }

    fn bytes_of_cnv_pvec_right(&self, cols: usize, size: usize) -> usize {
        BE::bytes_of_cnv_pvec_right(self.n(), cols, size)
    }
}

impl<BE: Backend> Convolution<BE> for Module<BE>
where
    BE: HalImpl<BE>,
{
    fn cnv_prepare_left_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize {
        <BE as HalImpl<BE>>::cnv_prepare_left_tmp_bytes(self, res_size, a_size)
    }
    fn cnv_prepare_left<R, A>(&self, res: &mut R, a: &A, mask: i64, scratch: &mut Scratch<BE>)
    where
        R: CnvPVecLToMut<BE> + ZnxInfos + ZnxViewMut<Scalar = BE::ScalarPrep>,
        A: VecZnxToRef + ZnxInfos,
    {
        <BE as HalImpl<BE>>::cnv_prepare_left(self, res, a, mask, scratch);
    }

    fn cnv_prepare_right_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize {
        <BE as HalImpl<BE>>::cnv_prepare_right_tmp_bytes(self, res_size, a_size)
    }
    fn cnv_prepare_right<R, A>(&self, res: &mut R, a: &A, mask: i64, scratch: &mut Scratch<BE>)
    where
        R: CnvPVecRToMut<BE> + ZnxInfos + ZnxViewMut<Scalar = BE::ScalarPrep>,
        A: VecZnxToRef + ZnxInfos,
    {
        <BE as HalImpl<BE>>::cnv_prepare_right(self, res, a, mask, scratch);
    }

    fn cnv_apply_dft_tmp_bytes(&self, res_size: usize, cnv_offset: usize, a_size: usize, b_size: usize) -> usize {
        <BE as HalImpl<BE>>::cnv_apply_dft_tmp_bytes(self, res_size, cnv_offset, a_size, b_size)
    }

    fn cnv_by_const_apply_tmp_bytes(&self, res_size: usize, cnv_offset: usize, a_size: usize, b_size: usize) -> usize {
        <BE as HalImpl<BE>>::cnv_by_const_apply_tmp_bytes(self, res_size, cnv_offset, a_size, b_size)
    }

    fn cnv_by_const_apply<R, A>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &[i64],
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxBigToMut<BE>,
        A: VecZnxToRef,
    {
        <BE as HalImpl<BE>>::cnv_by_const_apply(self, cnv_offset, res, res_col, a, a_col, b, scratch);
    }

    fn cnv_apply_dft<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxDftToMut<BE>,
        A: CnvPVecLToRef<BE>,
        B: CnvPVecRToRef<BE>,
    {
        <BE as HalImpl<BE>>::cnv_apply_dft(self, cnv_offset, res, res_col, a, a_col, b, b_col, scratch);
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize {
        <BE as HalImpl<BE>>::cnv_pairwise_apply_dft_tmp_bytes(self, res_size, cnv_offset, a_size, b_size)
    }

    fn cnv_pairwise_apply_dft<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        b: &B,
        i: usize,
        j: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxDftToMut<BE>,
        A: CnvPVecLToRef<BE>,
        B: CnvPVecRToRef<BE>,
    {
        <BE as HalImpl<BE>>::cnv_pairwise_apply_dft(self, cnv_offset, res, res_col, a, b, i, j, scratch);
    }

    fn cnv_tensor_r1_fused_apply_dft_tmp_bytes(
        &self,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <BE as HalImpl<BE>>::cnv_tensor_r1_fused_apply_dft_tmp_bytes(self, cnv_offset, res_size, a_size, b_size)
    }

    fn cnv_tensor_r1_fused_apply_dft<R0, R1, RP, A, B>(
        &self,
        cnv_offset: usize,
        res_diag_0: &mut R0,
        res_diag_1: &mut R1,
        res_pair: &mut RP,
        a: &A,
        b: &B,
        scratch: &mut Scratch<BE>,
    ) where
        R0: VecZnxDftToMut<BE>,
        R1: VecZnxDftToMut<BE>,
        RP: VecZnxDftToMut<BE>,
        A: CnvPVecLToRef<BE>,
        B: CnvPVecRToRef<BE>,
    {
        <BE as HalImpl<BE>>::cnv_tensor_r1_fused_apply_dft(self, cnv_offset, res_diag_0, res_diag_1, res_pair, a, b, scratch);
    }

    fn cnv_prepare_self_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize {
        <BE as HalImpl<BE>>::cnv_prepare_self_tmp_bytes(self, res_size, a_size)
    }

    fn cnv_prepare_self<L, R, A>(&self, left: &mut L, right: &mut R, a: &A, mask: i64, scratch: &mut Scratch<BE>)
    where
        L: CnvPVecLToMut<BE> + ZnxInfos + ZnxViewMut<Scalar = BE::ScalarPrep>,
        R: CnvPVecRToMut<BE> + ZnxInfos + ZnxViewMut<Scalar = BE::ScalarPrep>,
        A: VecZnxToRef + ZnxInfos,
    {
        <BE as HalImpl<BE>>::cnv_prepare_self(self, left, right, a, mask, scratch);
    }
}
