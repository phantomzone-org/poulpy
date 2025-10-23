use crate::{
    api::{
        ModuleN, ScratchTakeBasic, SvpApplyDftToDft, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare, VecZnxDftAddScaledInplace,
        VecZnxDftBytesOf,
    },
    layouts::{Backend, Module, Scratch, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, ZnxInfos, ZnxZero},
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
        + VecZnxDftAddScaledInplace<BE>,
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
        + VecZnxDftAddScaledInplace<BE>,
    Scratch<BE>: ScratchTakeBasic,
{
    fn convolution_tmp_bytes(&self, res_size: usize) -> usize {
        self.bytes_of_svp_ppol(1) + self.bytes_of_vec_znx_dft(1, res_size)
    }

    /// Evaluates a bivariate convolution over Z[X, Y] / (X^N + 1) where Y = 2^-K
    /// and scales the result by 2^{res_scale * K}
    ///
    /// # Example
    /// a = [a00, a10, a20, a30] = (a00 * 2^-K + a01 * 2^-2K) + (a10 * 2^-K + a11 * 2^-2K) * X ...
    ///     [a01, a11, a21, a31]
    ///
    /// b = [b00, b10, b20, b30] = (b00 * 2^-K + b01 * 2^-2K) + (b10 * 2^-K + b11 * 2^-2K) * X ...
    ///     [b01, b11, b21, b31]
    ///
    /// If res_scale = 0:
    /// res = [  0,   0,   0,   0] = (r01 * 2^-2K + r02 * 2^-3K + r03 * 2^-4K + r04 * 2^-5K) + ...
    ///       [r01, r11, r21, r31]
    ///       [r02, r12, r22, r32]
    ///       [r03, r13, r23, r33]
    ///       [r04, r14, r24, r34]
    ///
    /// If res_scale = 1:
    /// res = [r01, r11, r21, r31] = (r01 * 2^-K + r02 * 2^-2K + r03 * 2^-3K + r04 * 2^-4K + r05 * 2^-5K) + ...
    ///       [r02, r12, r22, r32]
    ///       [r03, r13, r23, r33]
    ///       [r04, r14, r24, r34]
    ///       [r05, r15, r25, r35]
    ///
    /// If res_scale = -1:
    /// res = [  0,   0,   0,   0] = (r01 * 2^-3K + r02 * 2^-4K + r03 * 2^-5K) + ...
    ///       [  0,   0,   0,   0]
    ///       [r01, r11, r21, r31]
    ///       [r02, r12, r22, r32]
    ///       [r03, r13, r23, r33]
    ///
    /// If res.size() < a.size() + b.size() + 1 + res_scale, result is truncated accordingly in the Y dimension.
    fn convolution<R, A, B>(&self, res: &mut R, res_scale: i64, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: VecZnxDftToMut<BE>,
        A: VecZnxToRef,
        B: VecZnxDftToRef<BE>,
    {
        let res: &mut crate::layouts::VecZnxDft<&mut [u8], BE> = &mut res.to_mut();
        let a: &crate::layouts::VecZnx<&[u8]> = &a.to_ref();
        let b: &crate::layouts::VecZnxDft<&[u8], BE> = &b.to_ref();

        assert!(res.cols() >= a.cols() + b.cols() - 1);

        res.zero();

        let (mut ppol, scratch_1) = scratch.take_svp_ppol(self, 1);
        let (mut res_tmp, _) = scratch_1.take_vec_znx_dft(self, 1, res.size());

        for a_col in 0..a.cols() {
            for a_limb in 0..a.size() {
                // Prepares the j-th limb of the i-th col of A
                self.svp_prepare(&mut ppol, 0, &a.as_scalar_znx_ref(a_col, a_limb), 0);

                for b_col in 0..b.cols() {
                    // Multiplies with the i-th col of B
                    self.svp_apply_dft_to_dft(&mut res_tmp, 0, &ppol, 0, b, b_col);

                    // Adds on the [a_col + b_col] of res, scaled by 2^{-(a_limb + 1) * Base2K}
                    self.vec_znx_dft_add_scaled_inplace(
                        res,
                        a_col + b_col,
                        &res_tmp,
                        0,
                        -(1 + a_limb as i64) + res_scale,
                    );
                }
            }
        }
    }
}
