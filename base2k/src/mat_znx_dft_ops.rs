use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::ffi::vmp;
use crate::znx_base::{ZnxInfos, ZnxLayout};
use crate::{Backend, FFT64, MatZnxDft, Module, VecZnxDft, assert_alignement};

/// This trait implements methods for vector matrix product,
/// that is, multiplying a [VecZnx] with a [MatZnxDft].
pub trait MatZnxDftOps<B: Backend> {
    /// Allocates a new [MatZnxDft] with the given number of rows and columns.
    ///
    /// # Arguments
    ///
    /// * `rows`: number of rows (number of [VecZnxDft]).
    /// * `size`: number of size (number of size of each [VecZnxDft]).
    fn new_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxDft<B>;

    fn bytes_of_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;

    fn new_mat_znx_dft_from_bytes(
        &self,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> MatZnxDft<FFT64>;

    fn new_mat_znx_dft_from_bytes_borrow(
        &self,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: &mut [u8],
    ) -> MatZnxDft<FFT64>;

    /// Prepares the ith-row of [MatZnxDft] from a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `b`: [MatZnxDft] on which the values are encoded.
    /// * `a`: the [VecZnxDft] to encode on the [MatZnxDft].
    /// * `row_i`: the index of the row to prepare.
    ///
    /// The size of buf can be obtained with [MatZnxDftOps::vmp_prepare_tmp_bytes].
    fn vmp_prepare_row(&self, b: &mut MatZnxDft<B>, b_row: usize, b_col_in: usize, a: &VecZnxDft<B>);

    /// Extracts the ith-row of [MatZnxDft] into a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `b`: the [VecZnxDft] to on which to extract the row of the [MatZnxDft].
    /// * `a`: [MatZnxDft] on which the values are encoded.
    /// * `row_i`: the index of the row to extract.
    fn vmp_extract_row(&self, b: &mut VecZnxDft<B>, a: &MatZnxDft<B>, a_row: usize, a_col_in: usize);

    /// Returns the size of the stratch space necessary for [MatZnxDftOps::vmp_apply_dft_to_dft].
    ///
    /// # Arguments
    ///
    /// * `c_size`: number of size of the output [VecZnxDft].
    /// * `a_size`: number of size of the input [VecZnxDft].
    /// * `rows`: number of rows of the input [MatZnxDft].
    /// * `size`: number of size of the input [MatZnxDft].
    fn vmp_apply_tmp_bytes(
        &self,
        c_cols: usize,
        c_size: usize,
        a_cols: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;

    /// Applies the vector matrix product [VecZnxDft] x [MatZnxDft].
    /// The size of `buf` is given by [MatZnxDftOps::vmp_apply_tmp_bytes].
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
    /// * `buf`: scratch space, the size can be obtained with [MatZnxDftOps::vmp_apply_tmp_bytes].
    fn vmp_apply(&self, c: &mut VecZnxDft<B>, a: &VecZnxDft<B>, b: &MatZnxDft<B>, buf: &mut [u8]);
}

impl MatZnxDftOps<FFT64> for Module<FFT64> {
    fn new_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxDft<FFT64> {
        MatZnxDft::<FFT64>::new(self, rows, cols_in, cols_out, size)
    }

    fn bytes_of_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        MatZnxDft::<FFT64>::bytes_of(self, rows, cols_in, cols_out, size)
    }

    fn new_mat_znx_dft_from_bytes(
        &self,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> MatZnxDft<FFT64> {
        MatZnxDft::<FFT64>::from_bytes(self, rows, cols_in, cols_out, size, bytes)
    }

    fn new_mat_znx_dft_from_bytes_borrow(
        &self,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: &mut [u8],
    ) -> MatZnxDft<FFT64> {
        MatZnxDft::<FFT64>::from_bytes_borrow(self, rows, cols_in, cols_out, size, bytes)
    }

    fn vmp_prepare_row(&self, b: &mut MatZnxDft<FFT64>, b_row: usize, b_col_in: usize, a: &VecZnxDft<FFT64>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(b.n(), self.n());
            assert_eq!(a.n(), self.n());
            assert_eq!(
                a.cols(),
                b.cols_out(),
                "a.cols(): {} != b.cols_out(): {}",
                a.cols(),
                b.cols_out()
            );
            assert!(
                b_row < b.rows(),
                "b_row: {} >= b.rows(): {}",
                b_row,
                b.rows()
            );
            assert!(
                b_col_in < b.cols_in(),
                "b_col_in: {} >= b.cols_in(): {}",
                b_col_in,
                b.cols_in()
            );
            assert_eq!(
                b.size(),
                a.size(),
                "b.size(): {} != a.size(): {}",
                b.size(),
                a.size()
            );
        }

        unsafe {
            vmp::vmp_prepare_row_dft(
                self.ptr,
                b.as_mut_ptr() as *mut vmp::vmp_pmat_t,
                a.as_ptr() as *const vec_znx_dft_t,
                (b_row * b.cols_in() + b_col_in) as u64,
                (b.rows() * b.cols_in()) as u64,
                (b.size() * b.cols_out()) as u64,
            );
        }
    }

    fn vmp_extract_row(&self, b: &mut VecZnxDft<FFT64>, a: &MatZnxDft<FFT64>, a_row: usize, a_col_in: usize) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(b.n(), self.n());
            assert_eq!(a.n(), self.n());
            assert_eq!(
                b.cols(),
                a.cols_out(),
                "b.cols(): {} != a.cols_out(): {}",
                b.cols(),
                a.cols_out()
            );
            assert!(
                a_row < a.rows(),
                "a_row: {} >= a.rows(): {}",
                a_row,
                a.rows()
            );
            assert!(
                a_col_in < a.cols_in(),
                "a_col_in: {} >= a.cols_in(): {}",
                a_col_in,
                a.cols_in()
            );
            assert_eq!(
                b.size(),
                a.size(),
                "b.size(): {} != a.size(): {}",
                b.size(),
                a.size()
            );
        }
        unsafe {
            vmp::vmp_extract_row_dft(
                self.ptr,
                b.as_mut_ptr() as *mut vec_znx_dft_t,
                a.as_ptr() as *const vmp::vmp_pmat_t,
                (a_row * a.cols_in() + a_col_in) as u64,
                (a.rows() * a.cols_in()) as u64,
                (a.size() * a.cols_out()) as u64,
            );
        }
    }

    fn vmp_apply_tmp_bytes(
        &self,
        res_cols: usize,
        res_size: usize,
        a_size: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        unsafe {
            vmp::vmp_apply_dft_to_dft_tmp_bytes(
                self.ptr,
                (res_size * res_cols) as u64,
                (a_size * a_cols) as u64,
                (b_rows * b_cols_in) as u64,
                (b_size * b_cols_out) as u64,
            ) as usize
        }
    }

    fn vmp_apply(&self, res: &mut VecZnxDft<FFT64>, a: &VecZnxDft<FFT64>, b: &MatZnxDft<FFT64>, tmp_bytes: &mut [u8]) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(a.n(), self.n());
            assert_eq!(
                res.cols(),
                b.cols_out(),
                "res.cols(): {} != b.cols_out: {}",
                res.cols(),
                b.cols_out()
            );
            assert_eq!(
                a.cols(),
                b.cols_in(),
                "a.cols(): {} != b.cols_in: {}",
                a.cols(),
                b.cols_in()
            );
            assert!(
                tmp_bytes.len()
                    >= self.vmp_apply_tmp_bytes(
                        res.cols(),
                        res.size(),
                        a.cols(),
                        a.size(),
                        b.rows(),
                        b.cols_in(),
                        b.cols_out(),
                        b.size()
                    )
            );
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft_to_dft(
                self.ptr,
                res.as_mut_ptr() as *mut vec_znx_dft_t,
                (res.size() * res.cols()) as u64,
                a.as_ptr() as *const vec_znx_dft_t,
                (a.size() * a.cols()) as u64,
                b.as_ptr() as *const vmp::vmp_pmat_t,
                (b.rows() * b.cols_in()) as u64,
                (b.size() * b.cols_out()) as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Encoding, FFT64, MatZnxDft, MatZnxDftOps, Module, Sampling, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps,
        VecZnxOps, ZnxInfos, alloc_aligned, znx_base::ZnxLayout,
    };
    use sampling::source::Source;

    #[test]
    fn vmp_prepare_row() {
        let module: Module<FFT64> = Module::<FFT64>::new(16);
        let log_base2k: usize = 8;
        let mat_rows: usize = 4;
        let mat_cols_in: usize = 2;
        let mat_cols_out: usize = 2;
        let mat_size: usize = 5;
        let mut a: VecZnx = module.new_vec_znx(mat_cols_out, mat_size);
        let mut a_dft: VecZnxDft<FFT64> = module.new_vec_znx_dft(mat_cols_out, mat_size);
        let mut b_dft: VecZnxDft<FFT64> = module.new_vec_znx_dft(mat_cols_out, mat_size);
        let mut mat: MatZnxDft<FFT64> = module.new_mat_znx_dft(mat_rows, mat_cols_in, mat_cols_out, mat_size);

        for col_in in 0..mat_cols_in {
            for row_i in 0..mat_rows {
                let mut source: Source = Source::new([0u8; 32]);
                (0..mat_cols_out).for_each(|col_out| {
                    module.fill_uniform(log_base2k, &mut a, col_out, mat_size, &mut source);
                    module.vec_znx_dft(&mut a_dft, col_out, &a, col_out);
                });
                module.vmp_prepare_row(&mut mat, row_i, col_in, &a_dft);
                module.vmp_extract_row(&mut b_dft, &mat, row_i, col_in);
                assert_eq!(a_dft.raw(), b_dft.raw());
            }
        }

        module.free();
    }

    #[test]
    fn vmp_apply() {
        let log_n: i32 = 5;
        let n: usize = 1 << log_n;

        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let log_base2k: usize = 15;
        let a_size: usize = 5;
        let mat_size: usize = 6;
        let res_size: usize = 5;

        [1, 2].iter().for_each(|in_cols| {
            [1, 2].iter().for_each(|out_cols| {
                let a_cols: usize = *in_cols;
                let res_cols: usize = *out_cols;

                let mat_rows: usize = a_size;
                let mat_cols_in: usize = a_cols;
                let mat_cols_out: usize = res_cols;
                let res_cols: usize = mat_cols_out;

                let mut tmp_bytes_vmp: Vec<u8> = alloc_aligned(
                    module.vmp_apply_tmp_bytes(
                        res_cols,
                        res_size,
                        a_cols,
                        a_size,
                        mat_rows,
                        mat_cols_in,
                        mat_cols_out,
                        mat_size,
                    ) | module.vec_znx_big_normalize_tmp_bytes(),
                );

                let mut a: VecZnx = module.new_vec_znx(a_cols, a_size);

                (0..a_cols).for_each(|i| {
                    a.at_mut(i, 2)[i + 1] = 1;
                });

                let mut mat_znx_dft: MatZnxDft<FFT64> = module.new_mat_znx_dft(mat_rows, mat_cols_in, mat_cols_out, mat_size);

                let mut c_dft: VecZnxDft<FFT64> = module.new_vec_znx_dft(mat_cols_out, mat_size);
                let mut c_big: VecZnxBig<FFT64> = module.new_vec_znx_big(mat_cols_out, mat_size);

                let mut tmp: VecZnx = module.new_vec_znx(mat_cols_out, mat_size);

                // Construts a [VecZnxMatDft] that performs cyclic rotations on each submatrix.
                (0..a.size()).for_each(|row_i| {
                    (0..mat_cols_in).for_each(|col_in_i| {
                        (0..mat_cols_out).for_each(|col_out_i| {
                            let idx = 1 + col_in_i * mat_cols_out + col_out_i;
                            tmp.at_mut(col_out_i, row_i)[idx] = 1 as i64; // X^{idx}
                            module.vec_znx_dft(&mut c_dft, col_out_i, &tmp, col_out_i);
                            tmp.at_mut(col_out_i, row_i)[idx] = 0 as i64;
                        });
                        module.vmp_prepare_row(&mut mat_znx_dft, row_i, col_in_i, &c_dft);
                    });
                });

                let mut a_dft: VecZnxDft<FFT64> = module.new_vec_znx_dft(a_cols, a_size);
                (0..a_cols).for_each(|i| {
                    module.vec_znx_dft(&mut a_dft, i, &a, i);
                });

                module.vmp_apply(&mut c_dft, &a_dft, &mat_znx_dft, &mut tmp_bytes_vmp);

                let mut res_have_vi64: Vec<i64> = vec![i64::default(); n];

                let mut res_have: VecZnx = module.new_vec_znx(res_cols, res_size);
                (0..mat_cols_out).for_each(|i| {
                    module.vec_znx_idft_tmp_a(&mut c_big, i, &mut c_dft, i);
                    module.vec_znx_big_normalize(log_base2k, &mut res_have, i, &c_big, i, &mut tmp_bytes_vmp);
                });

                (0..mat_cols_out).for_each(|col_i| {
                    let mut res_want_vi64: Vec<i64> = vec![i64::default(); n];
                    (0..a_cols).for_each(|i| {
                        res_want_vi64[(i + 1) + (1 + i * mat_cols_out + col_i)] = 1;
                    });
                    res_have.decode_vec_i64(col_i, log_base2k, log_base2k * 3, &mut res_have_vi64);
                    assert_eq!(res_have_vi64, res_want_vi64);
                });
            });
        });

        module.free();
    }
}
