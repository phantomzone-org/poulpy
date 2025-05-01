use crate::ffi::vec_znx_big::vec_znx_big_t;
use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::ffi::vmp;
use crate::znx_base::{ZnxInfos, ZnxLayout};
use crate::{Backend, FFT64, MatZnxDft, Module, VecZnx, VecZnxBig, VecZnxDft, ZnxAlloc, assert_alignement};

/// This trait implements methods for vector matrix product,
/// that is, multiplying a [VecZnx] with a [MatZnxDft].
pub trait MatZnxDftOps<B: Backend> {
    /// Allocates a new [MatZnxDft] with the given number of rows and columns.
    ///
    /// # Arguments
    ///
    /// * `rows`: number of rows (number of [VecZnxDft]).
    /// * `size`: number of size (number of size of each [VecZnxDft]).
    fn new_mat_znx_dft(&self, rows: usize, cols: usize, size: usize) -> MatZnxDft<B>;

    fn bytes_of_mat_znx_dft(&self, rows: usize, cols: usize, size: usize) -> usize;

    fn new_mat_znx_dft_from_bytes(&self, rows: usize, cols: usize, size: usize, bytes: Vec<u8>) -> MatZnxDft<FFT64>;

    fn new_mat_znx_dft_from_bytes_borrow(&self, rows: usize, cols: usize, size: usize, bytes: &mut [u8]) -> MatZnxDft<FFT64>;

    /// Returns the number of bytes needed as scratch space for [MatZnxDftOps::vmp_prepare_contiguous].
    ///
    /// # Arguments
    ///
    /// * `rows`: number of rows of the [MatZnxDft] used in [MatZnxDftOps::vmp_prepare_contiguous].
    /// * `size`: number of size of the [MatZnxDft] used in [MatZnxDftOps::vmp_prepare_contiguous].
    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols: usize, size: usize) -> usize;

    /// Prepares a [MatZnxDft] from a contiguous array of [i64].
    /// The helper struct [Matrix3D] can be used to contruct and populate
    /// the appropriate contiguous array.
    ///
    /// # Arguments
    ///
    /// * `b`: [MatZnxDft] on which the values are encoded.
    /// * `a`: the contiguous array of [i64] of the 3D matrix to encode on the [MatZnxDft].
    /// * `buf`: scratch space, the size of buf can be obtained with [MatZnxDftOps::vmp_prepare_tmp_bytes].
    fn vmp_prepare_contiguous(&self, b: &mut MatZnxDft<B>, a: &[i64], buf: &mut [u8]);

    /// Prepares the ith-row of [MatZnxDft] from a [VecZnx].
    ///
    /// # Arguments
    ///
    /// * `b`: [MatZnxDft] on which the values are encoded.
    /// * `a`: the vector of [VecZnx] to encode on the [MatZnxDft].
    /// * `row_i`: the index of the row to prepare.
    /// * `buf`: scratch space, the size of buf can be obtained with [MatZnxDftOps::vmp_prepare_tmp_bytes].
    ///
    /// The size of buf can be obtained with [MatZnxDftOps::vmp_prepare_tmp_bytes].
    fn vmp_prepare_row(&self, b: &mut MatZnxDft<B>, a: &[i64], row_i: usize, tmp_bytes: &mut [u8]);

    /// Extracts the ith-row of [MatZnxDft] into a [VecZnxBig].
    ///
    /// # Arguments
    ///
    /// * `b`: the [VecZnxBig] to on which to extract the row of the [MatZnxDft].
    /// * `a`: [MatZnxDft] on which the values are encoded.
    /// * `row_i`: the index of the row to extract.
    fn vmp_extract_row(&self, b: &mut VecZnxBig<B>, a: &MatZnxDft<B>, row_i: usize);

    /// Prepares the ith-row of [MatZnxDft] from a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `b`: [MatZnxDft] on which the values are encoded.
    /// * `a`: the [VecZnxDft] to encode on the [MatZnxDft].
    /// * `row_i`: the index of the row to prepare.
    ///
    /// The size of buf can be obtained with [MatZnxDftOps::vmp_prepare_tmp_bytes].
    fn vmp_prepare_row_dft(&self, b: &mut MatZnxDft<B>, a: &VecZnxDft<B>, row_i: usize);

    /// Extracts the ith-row of [MatZnxDft] into a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `b`: the [VecZnxDft] to on which to extract the row of the [MatZnxDft].
    /// * `a`: [MatZnxDft] on which the values are encoded.
    /// * `row_i`: the index of the row to extract.
    fn vmp_extract_row_dft(&self, b: &mut VecZnxDft<B>, row_i: usize, a: &MatZnxDft<B>);

    /// Returns the size of the stratch space necessary for [MatZnxDftOps::vmp_apply_dft].
    ///
    /// # Arguments
    ///
    /// * `c_size`: number of size of the output [VecZnxDft].
    /// * `a_size`: number of size of the input [VecZnx].
    /// * `rows`: number of rows of the input [MatZnxDft].
    /// * `size`: number of size of the input [MatZnxDft].
    fn vmp_apply_dft_tmp_bytes(&self, c_size: usize, a_size: usize, b_rows: usize, b_size: usize) -> usize;

    /// Applies the vector matrix product [VecZnxDft] x [MatZnxDft].
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [MatZnxDft].
    ///
    /// As such, given an input [VecZnx] of `i` size and a [MatZnxDft] of `i` rows and
    /// `j` size, the output is a [VecZnx] of `j` size.
    ///
    /// If there is a mismatch between the dimensions the largest valid ones are used.
    ///
    /// ```text
    /// |a b c d| x |e f g| = (a * |e f g| + b * |h i j| + c * |k l m|) = |n o p|
    ///             |h i j|
    ///             |k l m|
    /// ```
    /// where each element is a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `c`: the output of the vector matrix product, as a [VecZnxDft].
    /// * `a`: the left operand [VecZnx] of the vector matrix product.
    /// * `b`: the right operand [MatZnxDft] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [MatZnxDftOps::vmp_apply_dft_tmp_bytes].
    fn vmp_apply_dft(&self, c: &mut VecZnxDft<B>, a: &VecZnx, b: &MatZnxDft<B>, buf: &mut [u8]);

    /// Applies the vector matrix product [VecZnxDft] x [MatZnxDft] and adds on the receiver.
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [MatZnxDft].
    ///
    /// As such, given an input [VecZnx] of `i` size and a [MatZnxDft] of `i` rows and
    /// `j` size, the output is a [VecZnx] of `j` size.
    ///
    /// If there is a mismatch between the dimensions the largest valid ones are used.
    ///
    /// ```text
    /// |a b c d| x |e f g| = (a * |e f g| + b * |h i j| + c * |k l m|) = |n o p|
    ///             |h i j|
    ///             |k l m|
    /// ```
    /// where each element is a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `c`: the operand on which the output of the vector matrix product is added, as a [VecZnxDft].
    /// * `a`: the left operand [VecZnx] of the vector matrix product.
    /// * `b`: the right operand [MatZnxDft] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [MatZnxDftOps::vmp_apply_dft_tmp_bytes].
    fn vmp_apply_dft_add(&self, c: &mut VecZnxDft<B>, a: &VecZnx, b: &MatZnxDft<B>, buf: &mut [u8]);

    /// Returns the size of the stratch space necessary for [MatZnxDftOps::vmp_apply_dft_to_dft].
    ///
    /// # Arguments
    ///
    /// * `c_size`: number of size of the output [VecZnxDft].
    /// * `a_size`: number of size of the input [VecZnxDft].
    /// * `rows`: number of rows of the input [MatZnxDft].
    /// * `size`: number of size of the input [MatZnxDft].
    fn vmp_apply_dft_to_dft_tmp_bytes(&self, c_size: usize, a_size: usize, rows: usize, size: usize) -> usize;

    /// Applies the vector matrix product [VecZnxDft] x [MatZnxDft].
    /// The size of `buf` is given by [MatZnxDftOps::vmp_apply_dft_to_dft_tmp_bytes].
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [MatZnxDft].
    ///
    /// As such, given an input [VecZnx] of `i` size and a [MatZnxDft] of `i` rows and
    /// `j` size, the output is a [VecZnx] of `j` size.
    ///
    /// If there is a mismatch between the dimensions the largest valid ones are used.
    ///
    /// ```text
    /// |a b c d| x |e f g| = (a * |e f g| + b * |h i j| + c * |k l m|) = |n o p|
    ///             |h i j|
    ///             |k l m|
    /// ```
    /// where each element is a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `c`: the output of the vector matrix product, as a [VecZnxDft].
    /// * `a`: the left operand [VecZnxDft] of the vector matrix product.
    /// * `b`: the right operand [MatZnxDft] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [MatZnxDftOps::vmp_apply_dft_to_dft_tmp_bytes].
    fn vmp_apply_dft_to_dft(&self, c: &mut VecZnxDft<B>, a: &VecZnxDft<B>, b: &MatZnxDft<B>, buf: &mut [u8]);

    /// Applies the vector matrix product [VecZnxDft] x [MatZnxDft] and adds on top of the receiver instead of overwritting it.
    /// The size of `buf` is given by [MatZnxDftOps::vmp_apply_dft_to_dft_tmp_bytes].
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [MatZnxDft].
    ///
    /// As such, given an input [VecZnx] of `i` size and a [MatZnxDft] of `i` rows and
    /// `j` size, the output is a [VecZnx] of `j` size.
    ///
    /// If there is a mismatch between the dimensions the largest valid ones are used.
    ///
    /// ```text
    /// |a b c d| x |e f g| = (a * |e f g| + b * |h i j| + c * |k l m|) = |n o p|
    ///             |h i j|
    ///             |k l m|
    /// ```
    /// where each element is a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `c`: the operand on which the output of the vector matrix product is added, as a [VecZnxDft].
    /// * `a`: the left operand [VecZnxDft] of the vector matrix product.
    /// * `b`: the right operand [MatZnxDft] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [MatZnxDftOps::vmp_apply_dft_to_dft_tmp_bytes].
    fn vmp_apply_dft_to_dft_add(&self, c: &mut VecZnxDft<B>, a: &VecZnxDft<B>, b: &MatZnxDft<B>, buf: &mut [u8]);

    /// Applies the vector matrix product [VecZnxDft] x [MatZnxDft] in place.
    /// The size of `buf` is given by [MatZnxDftOps::vmp_apply_dft_to_dft_tmp_bytes].
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [MatZnxDft].
    ///
    /// As such, given an input [VecZnx] of `i` size and a [MatZnxDft] of `i` rows and
    /// `j` size, the output is a [VecZnx] of `j` size.
    ///
    /// If there is a mismatch between the dimensions the largest valid ones are used.
    ///
    /// ```text
    /// |a b c d| x |e f g| = (a * |e f g| + b * |h i j| + c * |k l m|) = |n o p|
    ///             |h i j|
    ///             |k l m|
    /// ```
    /// where each element is a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `b`: the input and output of the vector matrix product, as a [VecZnxDft].
    /// * `a`: the right operand [MatZnxDft] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [MatZnxDftOps::vmp_apply_dft_to_dft_tmp_bytes].
    fn vmp_apply_dft_to_dft_inplace(&self, b: &mut VecZnxDft<B>, a: &MatZnxDft<B>, buf: &mut [u8]);
}

impl MatZnxDftOps<FFT64> for Module<FFT64> {
    fn new_mat_znx_dft(&self, rows: usize, cols: usize, size: usize) -> MatZnxDft<FFT64> {
        MatZnxDft::<FFT64>::new(self, rows, cols, size)
    }

    fn bytes_of_mat_znx_dft(&self, rows: usize, cols: usize, size: usize) -> usize {
        MatZnxDft::<FFT64>::bytes_of(self, rows, cols, size)
    }

    fn new_mat_znx_dft_from_bytes(&self, rows: usize, cols: usize, size: usize, bytes: Vec<u8>) -> MatZnxDft<FFT64> {
        MatZnxDft::<FFT64>::from_bytes(self, rows, cols, size, bytes)
    }

    fn new_mat_znx_dft_from_bytes_borrow(&self, rows: usize, cols: usize, size: usize, bytes: &mut [u8]) -> MatZnxDft<FFT64> {
        MatZnxDft::<FFT64>::from_bytes_borrow(self, rows, cols, size, bytes)
    }

    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols: usize, size: usize) -> usize {
        unsafe { vmp::vmp_prepare_tmp_bytes(self.ptr, rows as u64, (size * cols) as u64) as usize }
    }

    fn vmp_prepare_contiguous(&self, b: &mut MatZnxDft<FFT64>, a: &[i64], tmp_bytes: &mut [u8]) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.len(), b.n() * b.poly_count());
            assert!(tmp_bytes.len() >= self.vmp_prepare_tmp_bytes(b.rows(), b.cols(), b.size()));
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_prepare_contiguous(
                self.ptr,
                b.as_mut_ptr() as *mut vmp::vmp_pmat_t,
                a.as_ptr(),
                b.rows() as u64,
                (b.size() * b.cols()) as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vmp_prepare_row(&self, b: &mut MatZnxDft<FFT64>, a: &[i64], row_i: usize, tmp_bytes: &mut [u8]) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.len(), b.size() * self.n() * b.cols());
            assert!(tmp_bytes.len() >= self.vmp_prepare_tmp_bytes(b.rows(), b.cols(), b.size()));
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_prepare_row(
                self.ptr,
                b.as_mut_ptr() as *mut vmp::vmp_pmat_t,
                a.as_ptr(),
                row_i as u64,
                b.rows() as u64,
                (b.size() * b.cols()) as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vmp_extract_row(&self, b: &mut VecZnxBig<FFT64>, a: &MatZnxDft<FFT64>, row_i: usize) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), b.n());
            assert_eq!(a.size(), b.size());
            assert_eq!(a.cols(), b.cols());
        }
        unsafe {
            vmp::vmp_extract_row(
                self.ptr,
                b.as_mut_ptr() as *mut vec_znx_big_t,
                a.as_ptr() as *const vmp::vmp_pmat_t,
                row_i as u64,
                a.rows() as u64,
                (a.size() * a.cols()) as u64,
            );
        }
    }

    fn vmp_prepare_row_dft(&self, b: &mut MatZnxDft<FFT64>, a: &VecZnxDft<FFT64>, row_i: usize) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), b.n());
            assert_eq!(a.size(), b.size());
        }
        unsafe {
            vmp::vmp_prepare_row_dft(
                self.ptr,
                b.as_mut_ptr() as *mut vmp::vmp_pmat_t,
                a.as_ptr() as *const vec_znx_dft_t,
                row_i as u64,
                b.rows() as u64,
                b.size() as u64,
            );
        }
    }

    fn vmp_extract_row_dft(&self, b: &mut VecZnxDft<FFT64>, row_i: usize, a: &MatZnxDft<FFT64>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), b.n());
            assert_eq!(a.size(), b.size());
        }
        unsafe {
            vmp::vmp_extract_row_dft(
                self.ptr,
                b.as_mut_ptr() as *mut vec_znx_dft_t,
                a.as_ptr() as *const vmp::vmp_pmat_t,
                row_i as u64,
                a.rows() as u64,
                a.size() as u64,
            );
        }
    }

    fn vmp_apply_dft_tmp_bytes(&self, res_size: usize, a_size: usize, b_rows: usize, b_size: usize) -> usize {
        unsafe {
            vmp::vmp_apply_dft_tmp_bytes(
                self.ptr,
                res_size as u64,
                a_size as u64,
                b_rows as u64,
                b_size as u64,
            ) as usize
        }
    }

    fn vmp_apply_dft(&self, c: &mut VecZnxDft<FFT64>, a: &VecZnx, b: &MatZnxDft<FFT64>, tmp_bytes: &mut [u8]) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_tmp_bytes(c.size(), a.size(), b.rows(), b.size()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft(
                self.ptr,
                c.as_mut_ptr() as *mut vec_znx_dft_t,
                c.size() as u64,
                a.as_ptr(),
                a.size() as u64,
                (a.n() * a.cols()) as u64,
                b.as_ptr() as *const vmp::vmp_pmat_t,
                b.rows() as u64,
                b.size() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vmp_apply_dft_add(&self, c: &mut VecZnxDft<FFT64>, a: &VecZnx, b: &MatZnxDft<FFT64>, tmp_bytes: &mut [u8]) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_tmp_bytes(c.size(), a.size(), b.rows(), b.size()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft_add(
                self.ptr,
                c.as_mut_ptr() as *mut vec_znx_dft_t,
                c.size() as u64,
                a.as_ptr(),
                a.size() as u64,
                (a.n() * a.size()) as u64,
                b.as_ptr() as *const vmp::vmp_pmat_t,
                b.rows() as u64,
                b.size() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vmp_apply_dft_to_dft_tmp_bytes(&self, res_size: usize, a_size: usize, gct_rows: usize, gct_size: usize) -> usize {
        unsafe {
            vmp::vmp_apply_dft_to_dft_tmp_bytes(
                self.ptr,
                res_size as u64,
                a_size as u64,
                gct_rows as u64,
                gct_size as u64,
            ) as usize
        }
    }

    fn vmp_apply_dft_to_dft(&self, c: &mut VecZnxDft<FFT64>, a: &VecZnxDft<FFT64>, b: &MatZnxDft<FFT64>, tmp_bytes: &mut [u8]) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_to_dft_tmp_bytes(c.size(), a.size(), b.rows(), b.size()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft_to_dft(
                self.ptr,
                c.as_mut_ptr() as *mut vec_znx_dft_t,
                c.size() as u64,
                a.as_ptr() as *const vec_znx_dft_t,
                a.size() as u64,
                b.as_ptr() as *const vmp::vmp_pmat_t,
                b.rows() as u64,
                b.size() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vmp_apply_dft_to_dft_add(
        &self,
        c: &mut VecZnxDft<FFT64>,
        a: &VecZnxDft<FFT64>,
        b: &MatZnxDft<FFT64>,
        tmp_bytes: &mut [u8],
    ) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_to_dft_tmp_bytes(c.size(), a.size(), b.rows(), b.size()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft_to_dft_add(
                self.ptr,
                c.as_mut_ptr() as *mut vec_znx_dft_t,
                c.size() as u64,
                a.as_ptr() as *const vec_znx_dft_t,
                a.size() as u64,
                b.as_ptr() as *const vmp::vmp_pmat_t,
                b.rows() as u64,
                b.size() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vmp_apply_dft_to_dft_inplace(&self, b: &mut VecZnxDft<FFT64>, a: &MatZnxDft<FFT64>, tmp_bytes: &mut [u8]) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_to_dft_tmp_bytes(b.size(), b.size(), a.rows(), a.size()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft_to_dft(
                self.ptr,
                b.as_mut_ptr() as *mut vec_znx_dft_t,
                b.size() as u64,
                b.as_ptr() as *mut vec_znx_dft_t,
                b.size() as u64,
                a.as_ptr() as *const vmp::vmp_pmat_t,
                a.rows() as u64,
                a.size() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        FFT64, MatZnxDft, MatZnxDftOps, Module, Sampling, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, VecZnxOps,
        alloc_aligned, znx_base::ZnxLayout,
    };
    use sampling::source::Source;

    #[test]
    fn vmp_prepare_row_dft() {
        let module: Module<FFT64> = Module::<FFT64>::new(32);
        let vpmat_rows: usize = 4;
        let vpmat_size: usize = 5;
        let log_base2k: usize = 8;
        let mut a: VecZnx = module.new_vec_znx(1, vpmat_size);
        let mut a_dft: VecZnxDft<FFT64> = module.new_vec_znx_dft(1, vpmat_size);
        let mut a_big: VecZnxBig<FFT64> = module.new_vec_znx_big(1, vpmat_size);
        let mut b_big: VecZnxBig<FFT64> = module.new_vec_znx_big(1, vpmat_size);
        let mut b_dft: VecZnxDft<FFT64> = module.new_vec_znx_dft(1, vpmat_size);
        let mut vmpmat_0: MatZnxDft<FFT64> = module.new_mat_znx_dft(vpmat_rows, 1, vpmat_size);
        let mut vmpmat_1: MatZnxDft<FFT64> = module.new_mat_znx_dft(vpmat_rows, 1, vpmat_size);

        let mut tmp_bytes: Vec<u8> = alloc_aligned(module.vmp_prepare_tmp_bytes(vpmat_rows, 1, vpmat_size));

        for row_i in 0..vpmat_rows {
            let mut source: Source = Source::new([0u8; 32]);
            module.fill_uniform(log_base2k, &mut a, 0, vpmat_size, &mut source);
            module.vec_znx_dft(&mut a_dft, 0, &a, 0);
            module.vmp_prepare_row(&mut vmpmat_0, &a.raw(), row_i, &mut tmp_bytes);

            // Checks that prepare(mat_znx_dft, a) = prepare_dft(mat_znx_dft, a_dft)
            module.vmp_prepare_row_dft(&mut vmpmat_1, &a_dft, row_i);
            assert_eq!(vmpmat_0.raw(), vmpmat_1.raw());

            // Checks that a_dft = extract_dft(prepare(mat_znx_dft, a), b_dft)
            module.vmp_extract_row_dft(&mut b_dft, row_i, &vmpmat_0);
            assert_eq!(a_dft.raw(), b_dft.raw());

            // Checks that a_big = extract(prepare_dft(mat_znx_dft, a_dft), b_big)
            module.vmp_extract_row(&mut b_big, &vmpmat_0, row_i);
            module.vec_znx_idft(&mut a_big, 0, &a_dft, 0, &mut tmp_bytes);
            assert_eq!(a_big.raw(), b_big.raw());
        }

        module.free();
    }
}
