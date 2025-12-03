use crate::layouts::{
    Backend, CnvPVecL, CnvPVecLToMut, CnvPVecLToRef, CnvPVecR, CnvPVecRToMut, CnvPVecRToRef, Scratch, VecZnxDftToMut,
    VecZnxToRef, ZnxInfos, ZnxViewMut,
};

pub trait CnvPVecAlloc<BE: Backend> {
    fn cnv_pvec_left_alloc(&self, cols: usize, size: usize) -> CnvPVecL<Vec<u8>, BE>;
    fn cnv_pvec_right_alloc(&self, cols: usize, size: usize) -> CnvPVecR<Vec<u8>, BE>;
}

pub trait CnvPVecBytesOf {
    fn bytes_of_cnv_pvec_left(&self, cols: usize, size: usize) -> usize;
    fn bytes_of_cnv_pvec_right(&self, cols: usize, size: usize) -> usize;
}

pub trait Convolution<BE: Backend> {
    fn cnv_prepare_left_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    fn cnv_prepare_left<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: CnvPVecLToMut<BE> + ZnxInfos + ZnxViewMut<Scalar = BE::ScalarPrep>,
        A: VecZnxToRef + ZnxInfos;

    fn cnv_prepare_right_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    fn cnv_prepare_right<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: CnvPVecRToMut<BE> + ZnxInfos + ZnxViewMut<Scalar = BE::ScalarPrep>,
        A: VecZnxToRef + ZnxInfos;

    fn cnv_apply_dft_tmp_bytes(&self, res_size: usize, res_offset: usize, a_size: usize, b_size: usize) -> usize;

    #[allow(clippy::too_many_arguments)]
    /// Evaluates a bivariate convolution over Z[X, Y] / (X^N + 1) where Y = 2^-K over the
    /// selected columsn and stores the result on the selected column, scaled by 2^{res_offset * Base2K}
    ///
    /// # Example
    /// a = [a00, a10, a20, a30] = (a00 + a01 * 2^-K) + (a10 + a11 * 2^-K) * X ...
    ///     [a01, a11, a21, a31]
    ///
    /// b = [b00, b10, b20, b30] = (b00 + b01 * 2^-K) + (b10 + b11 * 2^-K) * X ...
    ///     [b01, b11, b21, b31]
    ///
    /// If res_offset = 0:
    /// res = [r00, r10, r20, r30] = (r00 + r01 * 2^-K + r02 * 2^-2K + r03 * 2^-3K) + ... * X + ...
    ///       [r01, r11, r21, r31]
    ///       [r02, r12, r22, r32]
    ///       [r03, r13, r23, r33]
    ///
    /// If res_offset = 1:
    /// res = [r01, r11, r21, r31]  = (r01 + r02 * 2^-K + r03 * 2^-2K) + ... * X + ...
    ///       [r02, r12, r22, r32]
    ///       [r03, r13, r23, r33]
    ///       [  0,   0,   0 ,  0]
    ///
    /// If res.size() < a.size() + b.size() + k, result is truncated accordingly in the Y dimension.
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
        B: CnvPVecRToRef<BE>;

    fn cnv_pairwise_apply_dft_tmp_bytes(&self, res_size: usize, res_offset: usize, a_size: usize, b_size: usize) -> usize;

    #[allow(clippy::too_many_arguments)]
    /// Evaluates the bivariate pair-wise convolution res = (a[i] + a[j]) * (b[i] + b[j]).
    /// If i == j then calls [Convolution::cnv_apply_dft], i.e. res = a[i] * b[i].
    /// See [Convolution::cnv_apply_dft] for informations about the bivariate convolution.
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
        B: CnvPVecRToRef<BE>;
}
