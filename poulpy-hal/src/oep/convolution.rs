use crate::layouts::{
    Backend, CnvPVecL, CnvPVecLToMut, CnvPVecLToRef, CnvPVecR, CnvPVecRToMut, CnvPVecRToRef, Module, Scratch, VecZnxDftToMut,
    VecZnxToRef, ZnxInfos, ZnxViewMut,
};

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See the TODO reference implementation.
/// * See [crate::api::CnvPVecLAlloc] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait CnvPVecLAllocImpl<BE: Backend> {
    fn cnv_pvec_left_alloc_impl(n: usize, cols: usize, size: usize) -> CnvPVecL<Vec<u8>, BE>;
    fn cnv_pvec_right_alloc_impl(n: usize, cols: usize, size: usize) -> CnvPVecR<Vec<u8>, BE>;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See the TODO reference implementation.
/// * See [crate::api::CnvPVecLBytesOf] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait CnvPVecBytesOfImpl {
    fn bytes_of_cnv_pvec_left_impl(n: usize, cols: usize, size: usize) -> usize;
    fn bytes_of_cnv_pvec_right_impl(n: usize, cols: usize, size: usize) -> usize;
}

/// # THIS TRAIT IS AN OPEN EXTENSION POINT (unsafe)
/// * See the TODO reference implementation.
/// * See [crate::api::Convolution] for corresponding public API.
/// # Safety [crate::doc::backend_safety] for safety contract.
pub unsafe trait ConvolutionImpl<BE: Backend> {
    fn cnv_prepare_left_tmp_bytes_impl(module: &Module<BE>, res_size: usize, a_size: usize) -> usize;
    fn cnv_prepare_left_impl<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: CnvPVecLToMut<BE> + ZnxInfos + ZnxViewMut<Scalar = BE::ScalarPrep>,
        A: VecZnxToRef + ZnxInfos;
    fn cnv_prepare_right_tmp_bytes_impl(module: &Module<BE>, res_size: usize, a_size: usize) -> usize;
    fn cnv_prepare_right_impl<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: CnvPVecRToMut<BE> + ZnxInfos + ZnxViewMut<Scalar = BE::ScalarPrep>,
        A: VecZnxToRef + ZnxInfos;
    fn cnv_apply_dft_tmp_bytes_impl(
        module: &Module<BE>,
        res_size: usize,
        res_offset: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;
    fn cnv_apply_dft_impl<R, A, B>(
        module: &Module<BE>,
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
        B: CnvPVecRToRef<BE>;
}
