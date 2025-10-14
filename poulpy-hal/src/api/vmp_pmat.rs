use crate::layouts::{
    Backend, MatZnxToRef, Scratch, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, VmpPMatOwned, VmpPMatToMut, VmpPMatToRef,
};

pub trait VmpPMatAlloc<B: Backend> {
    fn vmp_pmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<B>;
}

pub trait VmpPMatBytesOf {
    fn bytes_of_vmp_pmat(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

pub trait VmpPMatFromBytes<B: Backend> {
    fn vmp_pmat_from_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize, bytes: Vec<u8>) -> VmpPMatOwned<B>;
}

pub trait VmpPrepareTmpBytes {
    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

pub trait VmpPrepare<B: Backend> {
    fn vmp_prepare<R, A>(&self, pmat: &mut R, mat: &A, scratch: &mut Scratch<B>)
    where
        R: VmpPMatToMut<B>,
        A: MatZnxToRef;
}

#[allow(clippy::too_many_arguments)]
pub trait VmpApplyDftTmpBytes {
    fn vmp_apply_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

pub trait VmpApplyDft<B: Backend> {
    fn vmp_apply_dft<R, A, C>(&self, res: &mut R, a: &A, pmat: &C, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef,
        C: VmpPMatToRef<B>;
}

#[allow(clippy::too_many_arguments)]
pub trait VmpApplyDftToDftTmpBytes {
    fn vmp_apply_dft_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

pub trait VmpApplyDftToDft<B: Backend> {
    /// Applies the vector matrix product [crate::layouts::VecZnxDft] x [crate::layouts::VmpPMat].
    ///
    /// A vector matrix product numerically equivalent to a sum of [crate::api::SvpApply],
    /// where each [crate::layouts::SvpPPol] is a limb of the input [crate::layouts::VecZnx] in DFT,
    /// and each vector a [crate::layouts::VecZnxDft] (row) of the [crate::layouts::VmpPMat].
    ///
    /// As such, given an input [crate::layouts::VecZnx] of `i` size and a [crate::layouts::VmpPMat] of `i` rows and
    /// `j` size, the output is a [crate::layouts::VecZnx] of `j` size.
    ///
    /// If there is a mismatch between the dimensions the largest valid ones are used.
    ///
    /// ```text
    /// |a b c d| x |e f g| = (a * |e f g| + b * |h i j| + c * |k l m|) = |n o p|
    ///             |h i j|
    ///             |k l m|
    /// ```
    /// where each element is a [crate::layouts::VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `c`: the output of the vector matrix product, as a [crate::layouts::VecZnxDft].
    /// * `a`: the left operand [crate::layouts::VecZnxDft] of the vector matrix product.
    /// * `b`: the right operand [crate::layouts::VmpPMat] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [VmpApplyTmpBytes::vmp_apply_tmp_bytes].
    fn vmp_apply_dft_to_dft<R, A, C>(&self, res: &mut R, a: &A, pmat: &C, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>;
}

#[allow(clippy::too_many_arguments)]
pub trait VmpApplyDftToDftAddTmpBytes {
    fn vmp_apply_dft_to_dft_add_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

pub trait VmpApplyDftToDftAdd<B: Backend> {
    fn vmp_apply_dft_to_dft_add<R, A, C>(&self, res: &mut R, a: &A, b: &C, limb_offset: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        C: VmpPMatToRef<B>;
}
