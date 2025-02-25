use crate::ffi::vmp;
use crate::{Infos, Module, VecZnxApi, VecZnxDft};

/// Vector Matrix Product Prepared Matrix: a vector of [VecZnx],
/// stored as a 3D matrix in the DFT domain in a single contiguous array.
/// Each row of the [VmpPMat] can be seen as a [VecZnxDft].
///
/// The backend array of [VmpPMat] is allocate in C,
/// and thus must be manually freed.
///
/// [VmpPMat] is used to permform a vector matrix product between a [VecZnx] and a [VmpPMat].
/// See the trait [VmpPMatOps] for additional information.
pub struct VmpPMat {
    /// The pointer to the C memory.
    pub data: *mut vmp::vmp_pmat_t,
    /// The number of [VecZnxDft].
    pub rows: usize,
    /// The number of cols in each [VecZnxDft].      
    pub cols: usize,
    /// The ring degree of each [VecZnxDft].      
    pub n: usize,
}

impl VmpPMat {
    /// Returns the pointer to the [vmp_pmat_t].
    pub fn data(&self) -> *mut vmp::vmp_pmat_t {
        self.data
    }

    /// Returns a copy of the backend array at index (i, j) of the [VmpPMat].
    /// When using [`crate::FFT64`] as backend, `T` should be [f64].
    /// When using [`crate::NTT120`] as backend, `T` should be [i64].
    ///
    /// # Arguments
    ///
    /// * `row`: row index (i).
    /// * `col`: col index (j).
    pub fn at<T: Default + Copy>(&self, row: usize, col: usize) -> Vec<T> {
        let mut res: Vec<T> = vec![T::default(); self.n];

        if self.n < 8 {
            res.copy_from_slice(
                &self.get_backend_array::<T>()[(row + col * self.rows()) * self.n()
                    ..(row + col * self.rows()) * (self.n() + 1)],
            );
        } else {
            (0..self.n >> 3).for_each(|blk| {
                res[blk * 8..(blk + 1) * 8].copy_from_slice(&self.get_array(row, col, blk)[..8]);
            });
        }

        res
    }

    /// When using [`crate::FFT64`] as backend, `T` should be [f64].
    /// When using [`crate::NTT120`] as backend, `T` should be [i64].
    fn get_array<T>(&self, row: usize, col: usize, blk: usize) -> &[T] {
        let nrows: usize = self.rows();
        let ncols: usize = self.cols();
        if col == (ncols - 1) && (ncols & 1 == 1) {
            &self.get_backend_array::<T>()[blk * nrows * ncols * 8 + col * nrows * 8 + row * 8..]
        } else {
            &self.get_backend_array::<T>()[blk * nrows * ncols * 8
                + (col / 2) * (2 * nrows) * 8
                + row * 2 * 8
                + (col % 2) * 8..]
        }
    }

    /// Returns a non-mutable reference of `T` of the entire contiguous array of the [VmpPMat].
    /// When using [`crate::FFT64`] as backend, `T` should be [f64].
    /// When using [`crate::NTT120`] as backend, `T` should be [i64].
    /// The length of the returned array is rows * cols * n.
    pub fn get_backend_array<T>(&self) -> &[T] {
        let ptr: *const T = self.data as *const T;
        let len: usize = (self.rows() * self.cols() * self.n() * 8) / std::mem::size_of::<T>();
        unsafe { &std::slice::from_raw_parts(ptr, len) }
    }
}

/// This trait implements methods for vector matrix product,
/// that is, multiplying a [VecZnx] with a [VmpPMat].
pub trait VmpPMatOps {
    /// Allocates a new [VmpPMat] with the given number of rows and columns.
    ///
    /// # Arguments
    ///
    /// * `rows`: number of rows (number of [VecZnxDft]).
    /// * `cols`: number of cols (number of cols of each [VecZnxDft]).
    fn new_vmp_pmat(&self, rows: usize, cols: usize) -> VmpPMat;

    /// Returns the number of bytes needed as scratch space for [VmpPMatOps::vmp_prepare_contiguous].
    ///
    /// # Arguments
    ///
    /// * `rows`: number of rows of the [VmpPMat] used in [VmpPMatOps::vmp_prepare_contiguous].
    /// * `cols`: number of cols of the [VmpPMat] used in [VmpPMatOps::vmp_prepare_contiguous].
    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols: usize) -> usize;

    /// Prepares a [VmpPMat] from a contiguous array of [i64].
    /// The helper struct [Matrix3D] can be used to contruct and populate
    /// the appropriate contiguous array.
    ///
    /// # Arguments
    ///
    /// * `b`: [VmpPMat] on which the values are encoded.
    /// * `a`: the contiguous array of [i64] of the 3D matrix to encode on the [VmpPMat].
    /// * `buf`: scratch space, the size of buf can be obtained with [VmpPMatOps::vmp_prepare_tmp_bytes].
    ///
    /// # Example
    /// ```
    /// use base2k::{Module, VmpPMat, VmpPMatOps, FFT64, Free};
    /// use std::cmp::min;
    ///
    /// let n: usize = 1024;
    /// let module = Module::new::<FFT64>(n);
    /// let rows = 5;
    /// let cols = 6;
    ///
    /// let mut b_mat: Vec<i64> = vec![0i64;n * cols * rows];
    ///
    /// let mut buf: Vec<u8> = vec![u8::default(); module.vmp_prepare_tmp_bytes(rows, cols)];
    ///
    /// let mut vmp_pmat: VmpPMat = module.new_vmp_pmat(rows, cols);
    /// module.vmp_prepare_contiguous(&mut vmp_pmat, &b_mat, &mut buf);
    ///
    /// vmp_pmat.free() // don't forget to free the memory once vmp_pmat is not needed anymore.
    /// ```
    fn vmp_prepare_contiguous(&self, b: &mut VmpPMat, a: &[i64], buf: &mut [u8]);

    /// Prepares a [VmpPMat] from a vector of [VecZnx].
    ///
    /// # Arguments
    ///
    /// * `b`: [VmpPMat] on which the values are encoded.
    /// * `a`: the vector of [VecZnx] to encode on the [VmpPMat].
    /// * `buf`: scratch space, the size of buf can be obtained with [VmpPMatOps::vmp_prepare_tmp_bytes].
    ///
    /// The size of buf can be obtained with [VmpPMatOps::vmp_prepare_tmp_bytes].
    ///
    /// # Example
    /// ```
    /// use base2k::{Module, FFT64, VmpPMat, VmpPMatOps, VecZnx, VecZnxApi, VecZnxOps, Free};
    /// use std::cmp::min;
    ///
    /// let n: usize = 1024;
    /// let module: Module = Module::new::<FFT64>(n);
    /// let rows: usize = 5;
    /// let cols: usize = 6;
    ///
    /// let mut vecznx: Vec<VecZnx>= Vec::new();
    /// (0..rows).for_each(|_|{
    ///     vecznx.push(module.new_vec_znx(cols));
    /// });
    ///
    /// let slices: Vec<&[i64]> = vecznx.iter().map(|v| v.data.as_slice()).collect();
    ///
    /// let mut buf: Vec<u8> = vec![u8::default(); module.vmp_prepare_tmp_bytes(rows, cols)];
    ///
    /// let mut vmp_pmat: VmpPMat = module.new_vmp_pmat(rows, cols);
    /// module.vmp_prepare_dblptr(&mut vmp_pmat, &slices, &mut buf);
    ///
    /// vmp_pmat.free();
    /// module.free();
    /// ```
    fn vmp_prepare_dblptr(&self, b: &mut VmpPMat, a: &[&[i64]], buf: &mut [u8]);

    /// Prepares the ith-row of [VmpPMat] from a vector of [VecZnx].
    ///
    /// # Arguments
    ///
    /// * `b`: [VmpPMat] on which the values are encoded.
    /// * `a`: the vector of [VecZnx] to encode on the [VmpPMat].
    /// * `row_i`: the index of the row to prepare.
    /// * `buf`: scratch space, the size of buf can be obtained with [VmpPMatOps::vmp_prepare_tmp_bytes].
    ///
    /// The size of buf can be obtained with [VmpPMatOps::vmp_prepare_tmp_bytes].
    /// /// # Example
    /// ```
    /// use base2k::{Module, FFT64, VmpPMat, VmpPMatOps, VecZnx, VecZnxApi, VecZnxOps, Free};
    /// use std::cmp::min;
    ///
    /// let n: usize = 1024;
    /// let module: Module = Module::new::<FFT64>(n);
    /// let rows: usize = 5;
    /// let cols: usize = 6;
    ///
    /// let vecznx = module.new_vec_znx(cols);
    ///
    /// let mut buf: Vec<u8> = vec![u8::default(); module.vmp_prepare_tmp_bytes(rows, cols)];
    ///
    /// let mut vmp_pmat: VmpPMat = module.new_vmp_pmat(rows, cols);
    /// module.vmp_prepare_row(&mut vmp_pmat, vecznx.raw(), 0, &mut buf);
    ///
    /// vmp_pmat.free();
    /// module.free();
    /// ```
    fn vmp_prepare_row(&self, b: &mut VmpPMat, a: &[i64], row_i: usize, tmp_bytes: &mut [u8]);

    /// Returns the size of the stratch space necessary for [VmpPMatOps::vmp_apply_dft].
    ///
    /// # Arguments
    ///
    /// * `c_cols`: number of cols of the output [VecZnxDft].
    /// * `a_cols`: number of cols of the input [VecZnx].
    /// * `rows`: number of rows of the input [VmpPMat].
    /// * `cols`: number of cols of the input [VmpPMat].
    fn vmp_apply_dft_tmp_bytes(
        &self,
        c_cols: usize,
        a_cols: usize,
        rows: usize,
        cols: usize,
    ) -> usize;

    /// Applies the vector matrix product [VecZnxDft] x [VmpPMat].
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [VmpPMat].
    ///
    /// As such, given an input [VecZnx] of `i` cols and a [VmpPMat] of `i` rows and
    /// `j` cols, the output is a [VecZnx] of `j` cols.
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
    /// * `b`: the right operand [VmpPMat] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [VmpPMatOps::vmp_apply_dft_tmp_bytes].
    ///
    /// # Example
    /// ```
    /// use base2k::{Module, VecZnx, VecZnxOps, VecZnxDft, VecZnxDftOps, VmpPMat, VmpPMatOps, FFT64, Free, VecZnxApi};
    ///
    /// let n = 1024;
    ///
    /// let module: Module = Module::new::<FFT64>(n);
    /// let cols: usize = 5;
    ///
    /// let rows: usize = cols;
    /// let cols: usize = cols + 1;
    /// let c_cols: usize = cols;
    /// let a_cols: usize = cols;
    /// let tmp_bytes: usize = module.vmp_apply_dft_tmp_bytes(c_cols, a_cols, rows, cols);
    ///
    /// let mut buf: Vec<u8> = vec![0; tmp_bytes];
    /// let mut vmp_pmat: VmpPMat = module.new_vmp_pmat(rows, cols);
    ///
    /// let a: VecZnx = module.new_vec_znx(cols);
    /// let mut c_dft: VecZnxDft = module.new_vec_znx_dft(cols);
    /// module.vmp_apply_dft(&mut c_dft, &a, &vmp_pmat, &mut buf);
    ///
    /// c_dft.free();
    /// vmp_pmat.free();
    /// module.free();
    /// ```
    fn vmp_apply_dft<T: VecZnxApi + Infos>(
        &self,
        c: &mut VecZnxDft,
        a: &T,
        b: &VmpPMat,
        buf: &mut [u8],
    );

    /// Returns the size of the stratch space necessary for [VmpPMatOps::vmp_apply_dft_to_dft].
    ///
    /// # Arguments
    ///
    /// * `c_cols`: number of cols of the output [VecZnxDft].
    /// * `a_cols`: number of cols of the input [VecZnxDft].
    /// * `rows`: number of rows of the input [VmpPMat].
    /// * `cols`: number of cols of the input [VmpPMat].
    fn vmp_apply_dft_to_dft_tmp_bytes(
        &self,
        c_cols: usize,
        a_cols: usize,
        rows: usize,
        cols: usize,
    ) -> usize;

    /// Applies the vector matrix product [VecZnxDft] x [VmpPMat].
    /// The size of `buf` is given by [VmpPMatOps::vmp_apply_dft_to_dft_tmp_bytes].
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [VmpPMat].
    ///
    /// As such, given an input [VecZnx] of `i` cols and a [VmpPMat] of `i` rows and
    /// `j` cols, the output is a [VecZnx] of `j` cols.
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
    /// * `b`: the right operand [VmpPMat] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [VmpPMatOps::vmp_apply_dft_to_dft_tmp_bytes].
    ///
    /// # Example
    /// ```
    /// use base2k::{Module, VecZnx, VecZnxDft, VecZnxDftOps, VmpPMat, VmpPMatOps, FFT64, Free};
    ///
    /// let n = 1024;
    ///
    /// let module: Module = Module::new::<FFT64>(n);
    /// let cols: usize = 5;
    ///
    /// let rows: usize = cols;
    /// let cols: usize = cols + 1;
    /// let c_cols: usize = cols;
    /// let a_cols: usize = cols;
    /// let tmp_bytes: usize = module.vmp_apply_dft_to_dft_tmp_bytes(c_cols, a_cols, rows, cols);
    ///
    /// let mut buf: Vec<u8> = vec![0; tmp_bytes];
    /// let mut vmp_pmat: VmpPMat = module.new_vmp_pmat(rows, cols);
    ///
    /// let a_dft: VecZnxDft = module.new_vec_znx_dft(cols);
    /// let mut c_dft: VecZnxDft = module.new_vec_znx_dft(cols);
    /// module.vmp_apply_dft_to_dft(&mut c_dft, &a_dft, &vmp_pmat, &mut buf);
    ///
    /// a_dft.free();
    /// c_dft.free();
    /// vmp_pmat.free();
    /// module.free();
    /// ```
    fn vmp_apply_dft_to_dft(&self, c: &mut VecZnxDft, a: &VecZnxDft, b: &VmpPMat, buf: &mut [u8]);

    /// Applies the vector matrix product [VecZnxDft] x [VmpPMat] in place.
    /// The size of `buf` is given by [VmpPMatOps::vmp_apply_dft_to_dft_tmp_bytes].
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [VmpPMat].
    ///
    /// As such, given an input [VecZnx] of `i` cols and a [VmpPMat] of `i` rows and
    /// `j` cols, the output is a [VecZnx] of `j` cols.
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
    /// * `a`: the right operand [VmpPMat] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [VmpPMatOps::vmp_apply_dft_to_dft_tmp_bytes].
    ///
    /// # Example
    /// ```
    /// use base2k::{Module, VecZnx, VecZnxOps, VecZnxDft, VmpPMat, VmpPMatOps, FFT64, Free, VecZnxApi, VecZnxDftOps};
    ///
    /// let n = 1024;
    ///
    /// let module: Module = Module::new::<FFT64>(n);
    /// let cols: usize = 5;
    ///
    /// let rows: usize = cols;
    /// let cols: usize = cols + 1;
    /// let tmp_bytes: usize = module.vmp_apply_dft_to_dft_tmp_bytes(cols, cols, rows, cols);
    ///
    /// let mut buf: Vec<u8> = vec![0; tmp_bytes];
    /// let a: VecZnx = module.new_vec_znx(cols);
    /// let mut vmp_pmat: VmpPMat = module.new_vmp_pmat(rows, cols);
    ///
    /// let mut c_dft: VecZnxDft = module.new_vec_znx_dft(cols);
    /// module.vmp_apply_dft_to_dft_inplace(&mut c_dft, &vmp_pmat, &mut buf);
    ///
    /// c_dft.free();
    /// vmp_pmat.free();
    /// module.free();
    /// ```
    fn vmp_apply_dft_to_dft_inplace(&self, b: &mut VecZnxDft, a: &VmpPMat, buf: &mut [u8]);
}

impl VmpPMatOps for Module {
    fn new_vmp_pmat(&self, rows: usize, cols: usize) -> VmpPMat {
        unsafe {
            VmpPMat {
                data: vmp::new_vmp_pmat(self.0, rows as u64, cols as u64),
                rows,
                cols,
                n: self.n(),
            }
        }
    }

    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols: usize) -> usize {
        unsafe { vmp::vmp_prepare_tmp_bytes(self.0, rows as u64, cols as u64) as usize }
    }

    fn vmp_prepare_contiguous(&self, b: &mut VmpPMat, a: &[i64], buf: &mut [u8]) {
        unsafe {
            vmp::vmp_prepare_contiguous(
                self.0,
                b.data(),
                a.as_ptr(),
                b.rows() as u64,
                b.cols() as u64,
                buf.as_mut_ptr(),
            );
        }
    }

    fn vmp_prepare_dblptr(&self, b: &mut VmpPMat, a: &[&[i64]], buf: &mut [u8]) {
        let ptrs: Vec<*const i64> = a.iter().map(|v| v.as_ptr()).collect();
        unsafe {
            vmp::vmp_prepare_dblptr(
                self.0,
                b.data(),
                ptrs.as_ptr(),
                b.rows() as u64,
                b.cols() as u64,
                buf.as_mut_ptr(),
            );
        }
    }

    fn vmp_prepare_row(&self, b: &mut VmpPMat, a: &[i64], row_i: usize, buf: &mut [u8]) {
        debug_assert!(a.len() == b.cols() * self.n());
        unsafe {
            vmp::vmp_prepare_row(
                self.0,
                b.data(),
                a.as_ptr(),
                row_i as u64,
                b.rows() as u64,
                b.cols() as u64,
                buf.as_mut_ptr(),
            );
        }
    }

    fn vmp_apply_dft_tmp_bytes(
        &self,
        res_cols: usize,
        a_cols: usize,
        gct_rows: usize,
        gct_cols: usize,
    ) -> usize {
        unsafe {
            vmp::vmp_apply_dft_tmp_bytes(
                self.0,
                res_cols as u64,
                a_cols as u64,
                gct_rows as u64,
                gct_cols as u64,
            ) as usize
        }
    }

    fn vmp_apply_dft<T: VecZnxApi + Infos>(
        &self,
        c: &mut VecZnxDft,
        a: &T,
        b: &VmpPMat,
        buf: &mut [u8],
    ) {
        unsafe {
            vmp::vmp_apply_dft(
                self.0,
                c.0,
                c.cols() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
                b.data(),
                b.rows() as u64,
                b.cols() as u64,
                buf.as_mut_ptr(),
            )
        }
    }

    fn vmp_apply_dft_to_dft_tmp_bytes(
        &self,
        res_cols: usize,
        a_cols: usize,
        gct_rows: usize,
        gct_cols: usize,
    ) -> usize {
        unsafe {
            vmp::vmp_apply_dft_to_dft_tmp_bytes(
                self.0,
                res_cols as u64,
                a_cols as u64,
                gct_rows as u64,
                gct_cols as u64,
            ) as usize
        }
    }

    fn vmp_apply_dft_to_dft(&self, c: &mut VecZnxDft, a: &VecZnxDft, b: &VmpPMat, buf: &mut [u8]) {
        unsafe {
            vmp::vmp_apply_dft_to_dft(
                self.0,
                c.0,
                c.cols() as u64,
                a.0,
                a.cols() as u64,
                b.data(),
                b.rows() as u64,
                b.cols() as u64,
                buf.as_mut_ptr(),
            )
        }
    }

    fn vmp_apply_dft_to_dft_inplace(&self, b: &mut VecZnxDft, a: &VmpPMat, buf: &mut [u8]) {
        unsafe {
            vmp::vmp_apply_dft_to_dft(
                self.0,
                b.0,
                b.cols() as u64,
                b.0,
                b.cols() as u64,
                a.data(),
                a.rows() as u64,
                a.cols() as u64,
                buf.as_mut_ptr(),
            )
        }
    }
}
