use crate::{
    api::{
        ModuleN, ScratchTakeBasic, SvpApplyDftToDft, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare, VecZnxDftAddScaledInplace,
        VecZnxDftBytesOf, VecZnxDftZero,
    },
    layouts::{Backend, Module, Scratch, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, ZnxInfos},
};

impl<BE: Backend> Convolution<BE> for Module<BE>
where
    Self: Sized
        + ModuleN
        + SvpPPolAlloc<BE>
        + SvpApplyDftToDft<BE>
        + SvpPrepare<BE>
        + SvpPPolBytesOf
        + VecZnxDftBytesOf
        + VecZnxDftAddScaledInplace<BE>
        + VecZnxDftZero<BE>,
    Scratch<BE>: ScratchTakeBasic,
{
}

pub trait Convolution<BE: Backend>
where
    Self: Sized
        + ModuleN
        + SvpPPolAlloc<BE>
        + SvpApplyDftToDft<BE>
        + SvpPrepare<BE>
        + SvpPPolBytesOf
        + VecZnxDftBytesOf
        + VecZnxDftAddScaledInplace<BE>
        + VecZnxDftZero<BE>,
    Scratch<BE>: ScratchTakeBasic,
{
    fn convolution_tmp_bytes(&self, b_size: usize) -> usize {
        self.bytes_of_svp_ppol(1) + self.bytes_of_vec_znx_dft(1, b_size)
    }

    fn bivariate_convolution_full<R, A, B>(&self, k: i64, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: VecZnxDftToMut<BE>,
        A: VecZnxToRef,
        B: VecZnxDftToRef<BE>,
    {
        let res: &mut crate::layouts::VecZnxDft<&mut [u8], BE> = &mut res.to_mut();
        let a: &crate::layouts::VecZnx<&[u8]> = &a.to_ref();
        let b: &crate::layouts::VecZnxDft<&[u8], BE> = &b.to_ref();

        let res_cols: usize = res.cols();
        let a_cols: usize = a.cols();
        let b_cols: usize = b.cols();

        assert!(res_cols >= a_cols + b_cols - 1);

        for res_col in 0..res_cols {
            let a_min: usize = res_col.saturating_sub(b_cols - 1);
            let a_max: usize = res_col.min(a_cols - 1);
            self.bivariate_convolution_single(k, res, res_col, a, a_min, b, res_col - a_min, scratch);
            for a_col in a_min + 1..a_max + 1 {
                self.bivariate_convolution_single_add(k, res, res_col, a, a_col, b, res_col - a_col, scratch);
            }
        }
    }

    /// Evaluates a bivariate convolution over Z[X, Y] / (X^N + 1) where Y = 2^-K over the
    /// selected columsn and stores the result on the selected column, scaled by 2^{k * Base2K}
    ///
    /// # Example
    /// a = [a00, a10, a20, a30] = (a00 * 2^-K + a01 * 2^-2K) + (a10 * 2^-K + a11 * 2^-2K) * X ...
    ///     [a01, a11, a21, a31]
    ///
    /// b = [b00, b10, b20, b30] = (b00 * 2^-K + b01 * 2^-2K) + (b10 * 2^-K + b11 * 2^-2K) * X ...
    ///     [b01, b11, b21, b31]
    ///
    /// If k = 0:
    /// res = [  0,   0,   0,   0] = (r01 * 2^-2K + r02 * 2^-3K + r03 * 2^-4K + r04 * 2^-5K) + ...
    ///       [r01, r11, r21, r31]
    ///       [r02, r12, r22, r32]
    ///       [r03, r13, r23, r33]
    ///       [r04, r14, r24, r34]
    ///
    /// If k = 1:
    /// res = [r01, r11, r21, r31] = (r01 * 2^-K + r02 * 2^-2K + r03 * 2^-3K + r04 * 2^-4K + r05 * 2^-5K) + ...
    ///       [r02, r12, r22, r32]
    ///       [r03, r13, r23, r33]
    ///       [r04, r14, r24, r34]
    ///       [r05, r15, r25, r35]
    ///
    /// If k = -1:
    /// res = [  0,   0,   0,   0] = (r01 * 2^-3K + r02 * 2^-4K + r03 * 2^-5K) + ...
    ///       [  0,   0,   0,   0]
    ///       [r01, r11, r21, r31]
    ///       [r02, r12, r22, r32]
    ///       [r03, r13, r23, r33]
    ///
    /// If res.size() < a.size() + b.size() + 1 + k, result is truncated accordingly in the Y dimension.
    fn bivariate_convolution_single_add<R, A, B>(
        &self,
        k: i64,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxDftToMut<BE>,
        A: VecZnxToRef,
        B: VecZnxDftToRef<BE>,
    {
        let res: &mut crate::layouts::VecZnxDft<&mut [u8], BE> = &mut res.to_mut();
        let a: &crate::layouts::VecZnx<&[u8]> = &a.to_ref();
        let b: &crate::layouts::VecZnxDft<&[u8], BE> = &b.to_ref();

        let (mut ppol, scratch_1) = scratch.take_svp_ppol(self, 1);
        let (mut res_tmp, _) = scratch_1.take_vec_znx_dft(self, 1, b.size());

        for a_limb in 0..a.size() {
            self.svp_prepare(&mut ppol, 0, &a.as_scalar_znx_ref(a_col, a_limb), 0);
            self.svp_apply_dft_to_dft(&mut res_tmp, 0, &ppol, 0, b, b_col);
            self.vec_znx_dft_add_scaled_inplace(res, res_col, &res_tmp, 0, -(1 + a_limb as i64) + k);
        }
    }

    fn bivariate_convolution_single<R, A, B>(
        &self,
        k: i64,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxDftToMut<BE>,
        A: VecZnxToRef,
        B: VecZnxDftToRef<BE>,
    {
        self.vec_znx_dft_zero(res, res_col);
        self.bivariate_convolution_single_add(k, res, res_col, a, a_col, b, b_col, scratch);
    }
}
