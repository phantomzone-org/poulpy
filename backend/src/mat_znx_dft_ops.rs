use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::ffi::vmp;
use crate::znx_base::{ZnxInfos, ZnxView, ZnxViewMut};
use crate::{
    Backend, FFT64, MatZnxDft, MatZnxDftOwned, MatZnxToMut, MatZnxToRef, Module, Scratch, VecZnxDft, VecZnxDftToMut,
    VecZnxDftToRef,
};

pub trait MatZnxDftAlloc<B: Backend> {
    /// Allocates a new [MatZnxDft] with the given number of rows and columns.
    ///
    /// # Arguments
    ///
    /// * `rows`: number of rows (number of [VecZnxDft]).
    /// * `size`: number of size (number of size of each [VecZnxDft]).
    fn new_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxDftOwned<B>;

    fn bytes_of_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;

    fn new_mat_znx_dft_from_bytes(
        &self,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> MatZnxDftOwned<B>;
}

pub trait MatZnxDftScratch {
    /// Returns the size of the stratch space necessary for [MatZnxDftOps::vmp_apply_dft_to_dft].
    fn vmp_apply_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

/// This trait implements methods for vector matrix product,
/// that is, multiplying a [VecZnx] with a [MatZnxDft].
pub trait MatZnxDftOps<BACKEND: Backend> {
    /// Prepares the ith-row of [MatZnxDft] from a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `res`: [MatZnxDft] on which the values are encoded.
    /// * `a`: the [VecZnxDft] to encode on the [MatZnxDft].
    /// * `row_i`: the index of the row to prepare.
    ///
    /// The size of buf can be obtained with [MatZnxDftOps::vmp_prepare_tmp_bytes].
    fn vmp_prepare_row<R, A>(&self, res: &mut R, res_row: usize, res_col_in: usize, a: &A)
    where
        R: MatZnxToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>;

    /// Extracts the ith-row of [MatZnxDft] into a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `res`: the [VecZnxDft] to on which to extract the row of the [MatZnxDft].
    /// * `a`: [MatZnxDft] on which the values are encoded.
    /// * `row_i`: the index of the row to extract.
    fn vmp_extract_row<R, A>(&self, res: &mut R, a: &A, a_row: usize, a_col_in: usize)
    where
        R: VecZnxDftToMut<FFT64>,
        A: MatZnxToRef<FFT64>;

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
    fn vmp_apply<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch)
    where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
        B: MatZnxToRef<FFT64>;

    // Same as [MatZnxDftOps::vmp_apply] except result is added on R instead of overwritting R.
    fn vmp_apply_add<R, A, B>(&self, res: &mut R, a: &A, b: &B, scale: usize, scratch: &mut Scratch)
    where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
        B: MatZnxToRef<FFT64>;
}

impl<B: Backend> MatZnxDftAlloc<B> for Module<B> {
    fn bytes_of_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        MatZnxDftOwned::bytes_of(self, rows, cols_in, cols_out, size)
    }

    fn new_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxDftOwned<B> {
        MatZnxDftOwned::new(self, rows, cols_in, cols_out, size)
    }

    fn new_mat_znx_dft_from_bytes(
        &self,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> MatZnxDftOwned<B> {
        MatZnxDftOwned::new_from_bytes(self, rows, cols_in, cols_out, size, bytes)
    }
}

impl<BACKEND: Backend> MatZnxDftScratch for Module<BACKEND> {
    fn vmp_apply_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        unsafe {
            vmp::vmp_apply_dft_to_dft_tmp_bytes(
                self.ptr,
                (res_size * b_cols_out) as u64,
                (a_size * b_cols_in) as u64,
                (b_rows * b_cols_in) as u64,
                (b_size * b_cols_out) as u64,
            ) as usize
        }
    }
}

impl MatZnxDftOps<FFT64> for Module<FFT64> {
    fn vmp_prepare_row<R, A>(&self, res: &mut R, res_row: usize, res_col_in: usize, a: &A)
    where
        R: MatZnxToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
    {
        let mut res: MatZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: VecZnxDft<&[u8], _> = a.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), self.n());
            assert_eq!(a.n(), self.n());
            assert_eq!(
                a.cols(),
                res.cols_out(),
                "a.cols(): {} != res.cols_out(): {}",
                a.cols(),
                res.cols_out()
            );
            assert!(
                res_row < res.rows(),
                "res_row: {} >= res.rows(): {}",
                res_row,
                res.rows()
            );
            assert!(
                res_col_in < res.cols_in(),
                "res_col_in: {} >= res.cols_in(): {}",
                res_col_in,
                res.cols_in()
            );
            assert_eq!(
                res.size(),
                a.size(),
                "res.size(): {} != a.size(): {}",
                res.size(),
                a.size()
            );
        }

        unsafe {
            vmp::vmp_prepare_row_dft(
                self.ptr,
                res.as_mut_ptr() as *mut vmp::vmp_pmat_t,
                a.as_ptr() as *const vec_znx_dft_t,
                (res_row * res.cols_in() + res_col_in) as u64,
                (res.rows() * res.cols_in()) as u64,
                (res.size() * res.cols_out()) as u64,
            );
        }
    }

    fn vmp_extract_row<R, A>(&self, res: &mut R, a: &A, a_row: usize, a_col_in: usize)
    where
        R: VecZnxDftToMut<FFT64>,
        A: MatZnxToRef<FFT64>,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: MatZnxDft<&[u8], _> = a.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), self.n());
            assert_eq!(a.n(), self.n());
            assert_eq!(
                res.cols(),
                a.cols_out(),
                "res.cols(): {} != a.cols_out(): {}",
                res.cols(),
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
                res.size(),
                a.size(),
                "res.size(): {} != a.size(): {}",
                res.size(),
                a.size()
            );
        }
        unsafe {
            vmp::vmp_extract_row_dft(
                self.ptr,
                res.as_mut_ptr() as *mut vec_znx_dft_t,
                a.as_ptr() as *const vmp::vmp_pmat_t,
                (a_row * a.cols_in() + a_col_in) as u64,
                (a.rows() * a.cols_in()) as u64,
                (a.size() * a.cols_out()) as u64,
            );
        }
    }

    fn vmp_apply<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch)
    where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
        B: MatZnxToRef<FFT64>,
    {
        let mut res: VecZnxDft<&mut [u8], _> = res.to_mut();
        let a: VecZnxDft<&[u8], _> = a.to_ref();
        let b: MatZnxDft<&[u8], _> = b.to_ref();

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
        }

        let (tmp_bytes, _) = scratch.tmp_slice(self.vmp_apply_tmp_bytes(
            res.size(),
            a.size(),
            b.rows(),
            b.cols_in(),
            b.cols_out(),
            b.size(),
        ));
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

    fn vmp_apply_add<R, A, B>(&self, res: &mut R, a: &A, b: &B, scale: usize, scratch: &mut Scratch)
    where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
        B: MatZnxToRef<FFT64>,
    {
        let mut res: VecZnxDft<&mut [u8], _> = res.to_mut();
        let a: VecZnxDft<&[u8], _> = a.to_ref();
        let b: MatZnxDft<&[u8], _> = b.to_ref();

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
        }

        let (tmp_bytes, _) = scratch.tmp_slice(self.vmp_apply_tmp_bytes(
            res.size(),
            a.size(),
            b.rows(),
            b.cols_in(),
            b.cols_out(),
            b.size(),
        ));
        unsafe {
            vmp::vmp_apply_dft_to_dft_add(
                self.ptr,
                res.as_mut_ptr() as *mut vec_znx_dft_t,
                (res.size() * res.cols()) as u64,
                a.as_ptr() as *const vec_znx_dft_t,
                (a.size() * a.cols()) as u64,
                b.as_ptr() as *const vmp::vmp_pmat_t,
                (b.rows() * b.cols_in()) as u64,
                (b.size() * b.cols_out()) as u64,
                (scale * b.cols_out()) as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }
}
#[cfg(test)]
mod tests {
    use crate::{
        Decoding, FFT64, FillUniform, MatZnxDft, MatZnxDftOps, Module, ScratchOwned, VecZnx, VecZnxAlloc, VecZnxBig,
        VecZnxBigAlloc, VecZnxBigOps, VecZnxBigScratch, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxOps, ZnxInfos, ZnxView,
        ZnxViewMut, ZnxZero,
    };
    use sampling::source::Source;

    use super::{MatZnxDftAlloc, MatZnxDftScratch};

    #[test]
    fn vmp_prepare_row() {
        let module: Module<FFT64> = Module::<FFT64>::new(16);
        let basek: usize = 8;
        let mat_rows: usize = 4;
        let mat_cols_in: usize = 2;
        let mat_cols_out: usize = 2;
        let mat_size: usize = 5;
        let mut a: VecZnx<Vec<u8>> = module.new_vec_znx(mat_cols_out, mat_size);
        let mut a_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(mat_cols_out, mat_size);
        let mut b_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(mat_cols_out, mat_size);
        let mut mat: MatZnxDft<Vec<u8>, FFT64> = module.new_mat_znx_dft(mat_rows, mat_cols_in, mat_cols_out, mat_size);

        for col_in in 0..mat_cols_in {
            for row_i in 0..mat_rows {
                let mut source: Source = Source::new([0u8; 32]);
                (0..mat_cols_out).for_each(|col_out| {
                    a.fill_uniform(basek, col_out, mat_size, &mut source);
                    module.vec_znx_dft(1, 0, &mut a_dft, col_out, &a, col_out);
                });
                module.vmp_prepare_row(&mut mat, row_i, col_in, &a_dft);
                module.vmp_extract_row(&mut b_dft, &mat, row_i, col_in);
                assert_eq!(a_dft.raw(), b_dft.raw());
            }
        }
    }

    #[test]
    fn vmp_apply() {
        let log_n: i32 = 5;
        let n: usize = 1 << log_n;

        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let basek: usize = 15;
        let a_size: usize = 5;
        let mat_size: usize = 6;
        let res_size: usize = a_size;

        [1, 2].iter().for_each(|in_cols| {
            [1, 2].iter().for_each(|out_cols| {
                let a_cols: usize = *in_cols;
                let res_cols: usize = *out_cols;

                let mat_rows: usize = a_size;
                let mat_cols_in: usize = a_cols;
                let mat_cols_out: usize = res_cols;

                let mut scratch: ScratchOwned = ScratchOwned::new(
                    module.vmp_apply_tmp_bytes(
                        res_size,
                        a_size,
                        mat_rows,
                        mat_cols_in,
                        mat_cols_out,
                        mat_size,
                    ) | module.vec_znx_big_normalize_tmp_bytes(),
                );

                let mut a: VecZnx<Vec<u8>> = module.new_vec_znx(a_cols, a_size);

                (0..a_cols).for_each(|i| {
                    a.at_mut(i, a_size - 1)[i + 1] = 1;
                });

                let mut mat_znx_dft: MatZnxDft<Vec<u8>, FFT64> =
                    module.new_mat_znx_dft(mat_rows, mat_cols_in, mat_cols_out, mat_size);

                let mut c_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(mat_cols_out, mat_size);
                let mut c_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(mat_cols_out, mat_size);

                let mut tmp: VecZnx<Vec<u8>> = module.new_vec_znx(mat_cols_out, mat_size);

                // Construts a [VecZnxMatDft] that performs cyclic rotations on each submatrix.
                (0..a.size()).for_each(|row_i| {
                    (0..mat_cols_in).for_each(|col_in_i| {
                        (0..mat_cols_out).for_each(|col_out_i| {
                            let idx = 1 + col_in_i * mat_cols_out + col_out_i;
                            tmp.at_mut(col_out_i, row_i)[idx] = 1 as i64; // X^{idx}
                            module.vec_znx_dft(1, 0, &mut c_dft, col_out_i, &tmp, col_out_i);
                            tmp.at_mut(col_out_i, row_i)[idx] = 0 as i64;
                        });
                        module.vmp_prepare_row(&mut mat_znx_dft, row_i, col_in_i, &c_dft);
                    });
                });

                let mut a_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(a_cols, a_size);
                (0..a_cols).for_each(|i| {
                    module.vec_znx_dft(1, 0, &mut a_dft, i, &a, i);
                });

                module.vmp_apply(&mut c_dft, &a_dft, &mat_znx_dft, scratch.borrow());

                let mut res_have_vi64: Vec<i64> = vec![i64::default(); n];

                let mut res_have: VecZnx<Vec<u8>> = module.new_vec_znx(res_cols, res_size);
                (0..mat_cols_out).for_each(|i| {
                    module.vec_znx_idft_tmp_a(&mut c_big, i, &mut c_dft, i);
                    module.vec_znx_big_normalize(basek, &mut res_have, i, &c_big, i, scratch.borrow());
                });

                (0..mat_cols_out).for_each(|col_i| {
                    let mut res_want_vi64: Vec<i64> = vec![i64::default(); n];
                    (0..a_cols).for_each(|i| {
                        res_want_vi64[(i + 1) + (1 + i * mat_cols_out + col_i)] = 1;
                    });
                    res_have.decode_vec_i64(col_i, basek, basek * a_size, &mut res_have_vi64);
                    assert_eq!(res_have_vi64, res_want_vi64);
                });
            });
        });
    }

    #[test]
    fn vmp_apply_add() {
        let log_n: i32 = 4;
        let n: usize = 1 << log_n;

        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let basek: usize = 8;
        let a_size: usize = 5;
        let mat_size: usize = 5;
        let res_size: usize = a_size;
        let mut source: Source = Source::new([0u8; 32]);

        [1, 2].iter().for_each(|in_cols| {
            [1, 2].iter().for_each(|out_cols| {
                (0..res_size).for_each(|shift| {
                    let a_cols: usize = *in_cols;
                    let res_cols: usize = *out_cols;

                    let mat_rows: usize = a_size;
                    let mat_cols_in: usize = a_cols;
                    let mat_cols_out: usize = res_cols;

                    let mut scratch: ScratchOwned = ScratchOwned::new(
                        module.vmp_apply_tmp_bytes(
                            res_size,
                            a_size,
                            mat_rows,
                            mat_cols_in,
                            mat_cols_out,
                            mat_size,
                        ) | module.vec_znx_big_normalize_tmp_bytes(),
                    );

                    let mut a: VecZnx<Vec<u8>> = module.new_vec_znx(a_cols, a_size);

                    (0..a_cols).for_each(|col_i| {
                        a.fill_uniform(basek, col_i, a.size(), &mut source);
                    });

                    let mut mat_znx_dft: MatZnxDft<Vec<u8>, FFT64> =
                        module.new_mat_znx_dft(mat_rows, mat_cols_in, mat_cols_out, mat_size);

                    let mut c_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(mat_cols_out, mat_size);
                    let mut c_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(mat_cols_out, mat_size);

                    let mut tmp: VecZnx<Vec<u8>> = module.new_vec_znx(mat_cols_out, mat_size);

                    // Construts a [VecZnxMatDft] that performs cyclic rotations on each submatrix.
                    (0..a.size()).for_each(|row_i| {
                        (0..mat_cols_in).for_each(|col_in_i| {
                            (0..mat_cols_out).for_each(|col_out_i| {
                                let idx: usize = 1 + col_in_i * mat_cols_out + col_out_i;
                                tmp.at_mut(col_out_i, row_i)[idx] = 1 as i64; // X^{idx}
                                module.vec_znx_dft(1, 0, &mut c_dft, col_out_i, &tmp, col_out_i);
                                tmp.at_mut(col_out_i, row_i)[idx] = 0 as i64;
                            });
                            module.vmp_prepare_row(&mut mat_znx_dft, row_i, col_in_i, &c_dft);
                        });
                    });

                    let mut a_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(a_cols, a_size);
                    (0..a_cols).for_each(|i| {
                        module.vec_znx_dft(1, 0, &mut a_dft, i, &a, i);
                    });

                    c_dft.zero();
                    (0..c_dft.cols()).for_each(|i| {
                        module.vec_znx_dft(1, 0, &mut c_dft, i, &a, 0);
                    });

                    module.vmp_apply_add(&mut c_dft, &a_dft, &mat_znx_dft, shift, scratch.borrow());

                    let mut res_have: VecZnx<Vec<u8>> = module.new_vec_znx(res_cols, mat_size);
                    (0..mat_cols_out).for_each(|i| {
                        module.vec_znx_idft_tmp_a(&mut c_big, i, &mut c_dft, i);
                        module.vec_znx_big_normalize(basek, &mut res_have, i, &c_big, i, scratch.borrow());
                    });

                    let mut res_want: VecZnx<Vec<u8>> = module.new_vec_znx(res_cols, mat_size);

                    // Equivalent to vmp_add & scale
                    module.vmp_apply(&mut c_dft, &a_dft, &mat_znx_dft, scratch.borrow());
                    (0..mat_cols_out).for_each(|i| {
                        module.vec_znx_idft_tmp_a(&mut c_big, i, &mut c_dft, i);
                        module.vec_znx_big_normalize(basek, &mut res_want, i, &c_big, i, scratch.borrow());
                    });
                    module.vec_znx_shift_inplace(
                        basek,
                        (shift * basek) as i64,
                        &mut res_want,
                        scratch.borrow(),
                    );
                    (0..res_cols).for_each(|i| {
                        module.vec_znx_add_inplace(&mut res_want, i, &a, 0);
                        module.vec_znx_normalize_inplace(basek, &mut res_want, i, scratch.borrow());
                    });

                    assert_eq!(res_want, res_have);
                });
            });
        });
    }

    #[test]
    fn vmp_apply_digits() {
        let log_n: i32 = 4;
        let n: usize = 1 << log_n;

        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let basek: usize = 8;
        let a_size: usize = 6;
        let mat_size: usize = 6;
        let res_size: usize = a_size;

        [1, 2].iter().for_each(|in_cols| {
            [1, 2].iter().for_each(|out_cols| {
                [1, 3, 6].iter().for_each(|digits| {
                    let mut source: Source = Source::new([0u8; 32]);

                    let a_cols: usize = *in_cols;
                    let res_cols: usize = *out_cols;

                    let mat_rows: usize = a_size;
                    let mat_cols_in: usize = a_cols;
                    let mat_cols_out: usize = res_cols;

                    let mut scratch: ScratchOwned = ScratchOwned::new(
                        module.vmp_apply_tmp_bytes(
                            res_size,
                            a_size,
                            mat_rows,
                            mat_cols_in,
                            mat_cols_out,
                            mat_size,
                        ) | module.vec_znx_big_normalize_tmp_bytes(),
                    );

                    let mut a: VecZnx<Vec<u8>> = module.new_vec_znx(a_cols, a_size);

                    (0..a_cols).for_each(|col_i| {
                        a.fill_uniform(basek, col_i, a.size(), &mut source);
                    });

                    let mut mat_znx_dft: MatZnxDft<Vec<u8>, FFT64> =
                        module.new_mat_znx_dft(mat_rows, mat_cols_in, mat_cols_out, mat_size);

                    let mut c_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(mat_cols_out, mat_size);
                    let mut c_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(mat_cols_out, mat_size);

                    let mut tmp: VecZnx<Vec<u8>> = module.new_vec_znx(mat_cols_out, mat_size);

                    let rows: usize = a.size() / digits;

                    let shift: usize = 1;

                    // Construts a [VecZnxMatDft] that performs cyclic rotations on each submatrix.
                    (0..rows).for_each(|row_i| {
                        (0..mat_cols_in).for_each(|col_in_i| {
                            (0..mat_cols_out).for_each(|col_out_i| {
                                let idx: usize = shift + col_in_i * mat_cols_out + col_out_i;
                                let limb: usize = (digits - 1) + row_i * digits;
                                tmp.at_mut(col_out_i, limb)[idx] = 1 as i64; // X^{idx}
                                module.vec_znx_dft(1, 0, &mut c_dft, col_out_i, &tmp, col_out_i);
                                tmp.at_mut(col_out_i, limb)[idx] = 0 as i64;
                            });
                            module.vmp_prepare_row(&mut mat_znx_dft, row_i, col_in_i, &c_dft);
                        });
                    });

                    let mut a_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(a_cols, (a_size + digits - 1) / digits);

                    (0..*digits).for_each(|di| {
                        (0..a_cols).for_each(|col_i| {
                            module.vec_znx_dft(digits - 1 - di, *digits, &mut a_dft, col_i, &a, col_i);
                        });

                        if di == 0 {
                            module.vmp_apply(&mut c_dft, &a_dft, &mat_znx_dft, scratch.borrow());
                        } else {
                            module.vmp_apply_add(&mut c_dft, &a_dft, &mat_znx_dft, di, scratch.borrow());
                        }
                    });

                    let mut res_have: VecZnx<Vec<u8>> = module.new_vec_znx(res_cols, mat_size);
                    (0..mat_cols_out).for_each(|i| {
                        module.vec_znx_idft_tmp_a(&mut c_big, i, &mut c_dft, i);
                        module.vec_znx_big_normalize(basek, &mut res_have, i, &c_big, i, scratch.borrow());
                    });

                    let mut res_want: VecZnx<Vec<u8>> = module.new_vec_znx(res_cols, mat_size);
                    let mut tmp: VecZnx<Vec<u8>> = module.new_vec_znx(res_cols, mat_size);
                    (0..res_cols).for_each(|col_i| {
                        (0..a_cols).for_each(|j| {
                            module.vec_znx_rotate(
                                (col_i + j * mat_cols_out + shift) as i64,
                                &mut tmp,
                                0,
                                &a,
                                j,
                            );
                            module.vec_znx_add_inplace(&mut res_want, col_i, &tmp, 0);
                        });
                        module.vec_znx_normalize_inplace(basek, &mut res_want, col_i, scratch.borrow());
                    });

                    assert_eq!(res_have, res_want)
                });
            });
        });
    }
}
