use crate::{
    api::{CnvPVecAlloc, CnvPVecBytesOf, Convolution},
    layouts::{
        Backend, CnvPVecL, CnvPVecLToMut, CnvPVecLToRef, CnvPVecR, CnvPVecRToMut, CnvPVecRToRef, Module, Scratch, VecZnxDftToMut,
        VecZnxToRef, ZnxInfos, ZnxViewMut,
    },
    oep::{CnvPVecBytesOfImpl, CnvPVecLAllocImpl, ConvolutionImpl},
};

impl<BE: Backend> CnvPVecAlloc<BE> for Module<BE>
where
    BE: CnvPVecLAllocImpl<BE>,
{
    fn cnv_pvec_left_alloc(&self, cols: usize, size: usize) -> CnvPVecL<Vec<u8>, BE> {
        BE::cnv_pvec_left_alloc_impl(self.n(), cols, size)
    }

    fn cnv_pvec_right_alloc(&self, cols: usize, size: usize) -> CnvPVecR<Vec<u8>, BE> {
        BE::cnv_pvec_right_alloc_impl(self.n(), cols, size)
    }
}

impl<BE: Backend> CnvPVecBytesOf for Module<BE>
where
    BE: CnvPVecBytesOfImpl,
{
    fn bytes_of_cnv_pvec_left(&self, cols: usize, size: usize) -> usize {
        BE::bytes_of_cnv_pvec_left_impl(self.n(), cols, size)
    }

    fn bytes_of_cnv_pvec_right(&self, cols: usize, size: usize) -> usize {
        BE::bytes_of_cnv_pvec_right_impl(self.n(), cols, size)
    }
}

impl<BE: Backend> Convolution<BE> for Module<BE>
where
    BE: ConvolutionImpl<BE>,
{
    fn cnv_prepare_left_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize {
        BE::cnv_prepare_left_tmp_bytes_impl(self, res_size, a_size)
    }
    fn cnv_prepare_left<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: CnvPVecLToMut<BE> + ZnxInfos + ZnxViewMut<Scalar = BE::ScalarPrep>,
        A: VecZnxToRef + ZnxInfos,
    {
        BE::cnv_prepare_left_impl(self, res, a, scratch);
    }

    fn cnv_prepare_right_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize {
        BE::cnv_prepare_right_tmp_bytes_impl(self, res_size, a_size)
    }
    fn cnv_prepare_right<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: CnvPVecRToMut<BE> + ZnxInfos + ZnxViewMut<Scalar = BE::ScalarPrep>,
        A: VecZnxToRef + ZnxInfos,
    {
        BE::cnv_prepare_right_impl(self, res, a, scratch);
    }

    fn cnv_apply_dft_tmp_bytes(&self, res_size: usize, res_offset: usize, a_size: usize, b_size: usize) -> usize {
        BE::cnv_apply_dft_tmp_bytes_impl(self, res_size, res_offset, a_size, b_size)
    }
    fn cnv_apply_dft<R, A, B>(
        &self,
        res: &mut R,
        res_offset: usize,
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
        BE::cnv_apply_dft_impl(self, res, res_offset, res_col, a, a_col, b, b_col, scratch);
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(&self, res_size: usize, res_offset: usize, a_size: usize, b_size: usize) -> usize {
        BE::cnv_pairwise_apply_dft_tmp_bytes(self, res_size, res_offset, a_size, b_size)
    }

    fn cnv_pairwise_apply_dft<R, A, B>(
        &self,
        res: &mut R,
        res_offset: usize,
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
        BE::cnv_pairwise_apply_dft_impl(self, res, res_offset, res_col, a, b, i, j, scratch);
    }
}
