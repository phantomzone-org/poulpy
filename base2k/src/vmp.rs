use crate::ffi::vec_znx_big::vec_znx_big_t;
use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::ffi::vmp::{self, vmp_pmat_t};
use crate::{BACKEND, Infos, LAYOUT, Module, VecZnx, VecZnxBig, VecZnxDft, alloc_aligned, assert_alignement};

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
    /// Raw data, is empty if borrowing scratch space.
    data: Vec<u8>,
    /// Pointer to data. Can point to scratch space.
    ptr: *mut u8,
    /// The number of [VecZnxDft].
    rows: usize,
    /// The number of cols in each [VecZnxDft].      
    cols: usize,
    /// The ring degree of each [VecZnxDft].      
    n: usize,
    /// The number of stacked [VmpPMat], must be a square.
    size: usize,
    /// The memory layout of the stacked [VmpPMat].
    layout: LAYOUT,
    /// The backend fft or ntt.
    backend: BACKEND,
}

impl Infos for VmpPMat {
    /// Returns the ring dimension of the [VmpPMat].
    fn n(&self) -> usize {
        self.n
    }

    fn log_n(&self) -> usize {
        (usize::BITS - (self.n() - 1).leading_zeros()) as _
    }

    fn size(&self) -> usize {
        self.size
    }

    fn layout(&self) -> LAYOUT {
        self.layout
    }

    /// Returns the number of rows (i.e. of [VecZnxDft]) of the [VmpPMat]
    fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of cols of the [VmpPMat].
    /// The number of cols refers to the number of cols  
    /// of each [VecZnxDft].
    /// This method is equivalent to [Self::cols].
    fn cols(&self) -> usize {
        self.cols
    }
}

impl VmpPMat {
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn borrowed(&self) -> bool {
        self.data.len() == 0
    }

    /// Returns a non-mutable reference of `T` of the entire contiguous array of the [VmpPMat].
    /// When using [`crate::FFT64`] as backend, `T` should be [f64].
    /// When using [`crate::NTT120`] as backend, `T` should be [i64].
    /// The length of the returned array is rows * cols * n.
    pub fn raw<T>(&self) -> &[T] {
        let ptr: *const T = self.ptr as *const T;
        let len: usize = (self.rows() * self.cols() * self.n() * 8) / std::mem::size_of::<T>();
        unsafe { &std::slice::from_raw_parts(ptr, len) }
    }

    /// Returns a non-mutable reference of `T` of the entire contiguous array of the [VmpPMat].
    /// When using [`crate::FFT64`] as backend, `T` should be [f64].
    /// When using [`crate::NTT120`] as backend, `T` should be [i64].
    /// The length of the returned array is rows * cols * n.
    pub fn raw_mut<T>(&self) -> &mut [T] {
        let ptr: *mut T = self.ptr as *mut T;
        let len: usize = (self.rows() * self.cols() * self.n() * 8) / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
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
        let mut res: Vec<T> = alloc_aligned(self.n);

        if self.n < 8 {
            res.copy_from_slice(
                &self.raw::<T>()[(row + col * self.rows()) * self.n()..(row + col * self.rows()) * (self.n() + 1)],
            );
        } else {
            (0..self.n >> 3).for_each(|blk| {
                res[blk * 8..(blk + 1) * 8].copy_from_slice(&self.at_block(row, col, blk)[..8]);
            });
        }

        res
    }

    /// When using [`crate::FFT64`] as backend, `T` should be [f64].
    /// When using [`crate::NTT120`] as backend, `T` should be [i64].
    fn at_block<T>(&self, row: usize, col: usize, blk: usize) -> &[T] {
        let nrows: usize = self.rows();
        let ncols: usize = self.cols();
        if col == (ncols - 1) && (ncols & 1 == 1) {
            &self.raw::<T>()[blk * nrows * ncols * 8 + col * nrows * 8 + row * 8..]
        } else {
            &self.raw::<T>()[blk * nrows * ncols * 8 + (col / 2) * (2 * nrows) * 8 + row * 2 * 8 + (col % 2) * 8..]
        }
    }

    fn backend(&self) -> BACKEND {
        self.backend
    }
}

/// This trait implements methods for vector matrix product,
/// that is, multiplying a [VecZnx] with a [VmpPMat].
pub trait VmpPMatOps {
    fn bytes_of_vmp_pmat(&self, size: usize, rows: usize, cols: usize) -> usize;

    /// Allocates a new [VmpPMat] with the given number of rows and columns.
    ///
    /// # Arguments
    ///
    /// * `rows`: number of rows (number of [VecZnxDft]).
    /// * `cols`: number of cols (number of cols of each [VecZnxDft]).
    fn new_vmp_pmat(&self, size: usize, rows: usize, cols: usize) -> VmpPMat;

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
    fn vmp_prepare_dblptr(&self, b: &mut VmpPMat, a: &[&[i64]], buf: &mut [u8]);

    /// Prepares the ith-row of [VmpPMat] from a [VecZnx].
    ///
    /// # Arguments
    ///
    /// * `b`: [VmpPMat] on which the values are encoded.
    /// * `a`: the vector of [VecZnx] to encode on the [VmpPMat].
    /// * `row_i`: the index of the row to prepare.
    /// * `buf`: scratch space, the size of buf can be obtained with [VmpPMatOps::vmp_prepare_tmp_bytes].
    ///
    /// The size of buf can be obtained with [VmpPMatOps::vmp_prepare_tmp_bytes].
    fn vmp_prepare_row(&self, b: &mut VmpPMat, a: &[i64], row_i: usize, tmp_bytes: &mut [u8]);

    /// Extracts the ith-row of [VmpPMat] into a [VecZnxBig].
    ///
    /// # Arguments
    ///
    /// * `b`: the [VecZnxBig] to on which to extract the row of the [VmpPMat].
    /// * `a`: [VmpPMat] on which the values are encoded.
    /// * `row_i`: the index of the row to extract.
    fn vmp_extract_row(&self, b: &mut VecZnxBig, a: &VmpPMat, row_i: usize);

    /// Prepares the ith-row of [VmpPMat] from a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `b`: [VmpPMat] on which the values are encoded.
    /// * `a`: the [VecZnxDft] to encode on the [VmpPMat].
    /// * `row_i`: the index of the row to prepare.
    ///
    /// The size of buf can be obtained with [VmpPMatOps::vmp_prepare_tmp_bytes].
    fn vmp_prepare_row_dft(&self, b: &mut VmpPMat, a: &VecZnxDft, row_i: usize);

    /// Extracts the ith-row of [VmpPMat] into a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `b`: the [VecZnxDft] to on which to extract the row of the [VmpPMat].
    /// * `a`: [VmpPMat] on which the values are encoded.
    /// * `row_i`: the index of the row to extract.
    fn vmp_extract_row_dft(&self, b: &mut VecZnxDft, a: &VmpPMat, row_i: usize);

    /// Returns the size of the stratch space necessary for [VmpPMatOps::vmp_apply_dft].
    ///
    /// # Arguments
    ///
    /// * `c_cols`: number of cols of the output [VecZnxDft].
    /// * `a_cols`: number of cols of the input [VecZnx].
    /// * `rows`: number of rows of the input [VmpPMat].
    /// * `cols`: number of cols of the input [VmpPMat].
    fn vmp_apply_dft_tmp_bytes(&self, c_cols: usize, a_cols: usize, rows: usize, cols: usize) -> usize;

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
    fn vmp_apply_dft(&self, c: &mut VecZnxDft, a: &VecZnx, b: &VmpPMat, buf: &mut [u8]);

    /// Applies the vector matrix product [VecZnxDft] x [VmpPMat] and adds on the receiver.
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
    /// * `c`: the operand on which the output of the vector matrix product is added, as a [VecZnxDft].
    /// * `a`: the left operand [VecZnx] of the vector matrix product.
    /// * `b`: the right operand [VmpPMat] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [VmpPMatOps::vmp_apply_dft_tmp_bytes].
    fn vmp_apply_dft_add(&self, c: &mut VecZnxDft, a: &VecZnx, b: &VmpPMat, buf: &mut [u8]);

    /// Returns the size of the stratch space necessary for [VmpPMatOps::vmp_apply_dft_to_dft].
    ///
    /// # Arguments
    ///
    /// * `c_cols`: number of cols of the output [VecZnxDft].
    /// * `a_cols`: number of cols of the input [VecZnxDft].
    /// * `rows`: number of rows of the input [VmpPMat].
    /// * `cols`: number of cols of the input [VmpPMat].
    fn vmp_apply_dft_to_dft_tmp_bytes(&self, c_cols: usize, a_cols: usize, rows: usize, cols: usize) -> usize;

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
    fn vmp_apply_dft_to_dft(&self, c: &mut VecZnxDft, a: &VecZnxDft, b: &VmpPMat, buf: &mut [u8]);

    /// Applies the vector matrix product [VecZnxDft] x [VmpPMat] and adds on top of the receiver instead of overwritting it.
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
    /// * `c`: the operand on which the output of the vector matrix product is added, as a [VecZnxDft].
    /// * `a`: the left operand [VecZnxDft] of the vector matrix product.
    /// * `b`: the right operand [VmpPMat] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [VmpPMatOps::vmp_apply_dft_to_dft_tmp_bytes].
    fn vmp_apply_dft_to_dft_add(&self, c: &mut VecZnxDft, a: &VecZnxDft, b: &VmpPMat, buf: &mut [u8]);

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
    fn vmp_apply_dft_to_dft_inplace(&self, b: &mut VecZnxDft, a: &VmpPMat, buf: &mut [u8]);
}

impl VmpPMatOps for Module {
    fn bytes_of_vmp_pmat(&self, size: usize, rows: usize, cols: usize) -> usize {
        unsafe { vmp::bytes_of_vmp_pmat(self.ptr, rows as u64, cols as u64) as usize * size }
    }

    fn new_vmp_pmat(&self, size: usize, rows: usize, cols: usize) -> VmpPMat {
        let mut data: Vec<u8> = alloc_aligned::<u8>(self.bytes_of_vmp_pmat(size, rows, cols));
        let ptr: *mut u8 = data.as_mut_ptr();
        VmpPMat {
            data: data,
            ptr: ptr,
            n: self.n(),
            size: size,
            layout: LAYOUT::COL,
            cols: cols,
            rows: rows,
            backend: self.backend(),
        }
    }

    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols: usize) -> usize {
        unsafe { vmp::vmp_prepare_tmp_bytes(self.ptr, rows as u64, cols as u64) as usize }
    }

    fn vmp_prepare_contiguous(&self, b: &mut VmpPMat, a: &[i64], tmp_bytes: &mut [u8]) {
        debug_assert_eq!(a.len(), b.n * b.rows * b.cols);
        debug_assert!(tmp_bytes.len() >= self.vmp_prepare_tmp_bytes(b.rows(), b.cols()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_prepare_contiguous(
                self.ptr,
                b.as_mut_ptr() as *mut vmp_pmat_t,
                a.as_ptr(),
                b.rows() as u64,
                b.cols() as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vmp_prepare_dblptr(&self, b: &mut VmpPMat, a: &[&[i64]], tmp_bytes: &mut [u8]) {
        let ptrs: Vec<*const i64> = a.iter().map(|v| v.as_ptr()).collect();
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(a.len(), b.rows);
            a.iter().for_each(|ai| {
                debug_assert_eq!(ai.len(), b.n * b.cols);
            });
            debug_assert!(tmp_bytes.len() >= self.vmp_prepare_tmp_bytes(b.rows(), b.cols()));
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_prepare_dblptr(
                self.ptr,
                b.as_mut_ptr() as *mut vmp_pmat_t,
                ptrs.as_ptr(),
                b.rows() as u64,
                b.cols() as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vmp_prepare_row(&self, b: &mut VmpPMat, a: &[i64], row_i: usize, tmp_bytes: &mut [u8]) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.len(), b.cols() * self.n());
            assert!(tmp_bytes.len() >= self.vmp_prepare_tmp_bytes(b.rows(), b.cols()));
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_prepare_row(
                self.ptr,
                b.as_mut_ptr() as *mut vmp_pmat_t,
                a.as_ptr(),
                row_i as u64,
                b.rows() as u64,
                b.cols() as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vmp_extract_row(&self, b: &mut VecZnxBig, a: &VmpPMat, row_i: usize) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), b.n());
            assert_eq!(a.cols(), b.cols());
        }
        unsafe {
            vmp::vmp_extract_row(
                self.ptr,
                b.ptr as *mut vec_znx_big_t,
                a.as_ptr() as *const vmp_pmat_t,
                row_i as u64,
                a.rows() as u64,
                a.cols() as u64,
            );
        }
    }

    fn vmp_prepare_row_dft(&self, b: &mut VmpPMat, a: &VecZnxDft, row_i: usize) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), b.n());
            assert_eq!(a.cols(), b.cols());
        }
        unsafe {
            vmp::vmp_prepare_row_dft(
                self.ptr,
                b.as_mut_ptr() as *mut vmp_pmat_t,
                a.ptr as *const vec_znx_dft_t,
                row_i as u64,
                b.rows() as u64,
                b.cols() as u64,
            );
        }
    }

    fn vmp_extract_row_dft(&self, b: &mut VecZnxDft, a: &VmpPMat, row_i: usize) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), b.n());
            assert_eq!(a.cols(), b.cols());
        }
        unsafe {
            vmp::vmp_extract_row_dft(
                self.ptr,
                b.ptr as *mut vec_znx_dft_t,
                a.as_ptr() as *const vmp_pmat_t,
                row_i as u64,
                a.rows() as u64,
                a.cols() as u64,
            );
        }
    }

    fn vmp_apply_dft_tmp_bytes(&self, res_cols: usize, a_cols: usize, gct_rows: usize, gct_cols: usize) -> usize {
        unsafe {
            vmp::vmp_apply_dft_tmp_bytes(
                self.ptr,
                res_cols as u64,
                a_cols as u64,
                gct_rows as u64,
                gct_cols as u64,
            ) as usize
        }
    }

    fn vmp_apply_dft(&self, c: &mut VecZnxDft, a: &VecZnx, b: &VmpPMat, tmp_bytes: &mut [u8]) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_tmp_bytes(c.cols(), a.cols(), b.rows(), b.cols()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft(
                self.ptr,
                c.ptr as *mut vec_znx_dft_t,
                c.cols() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
                b.as_ptr() as *const vmp_pmat_t,
                b.rows() as u64,
                b.cols() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vmp_apply_dft_add(&self, c: &mut VecZnxDft, a: &VecZnx, b: &VmpPMat, tmp_bytes: &mut [u8]) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_tmp_bytes(c.cols(), a.cols(), b.rows(), b.cols()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft_add(
                self.ptr,
                c.ptr as *mut vec_znx_dft_t,
                c.cols() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
                b.as_ptr() as *const vmp_pmat_t,
                b.rows() as u64,
                b.cols() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vmp_apply_dft_to_dft_tmp_bytes(&self, res_cols: usize, a_cols: usize, gct_rows: usize, gct_cols: usize) -> usize {
        unsafe {
            vmp::vmp_apply_dft_to_dft_tmp_bytes(
                self.ptr,
                res_cols as u64,
                a_cols as u64,
                gct_rows as u64,
                gct_cols as u64,
            ) as usize
        }
    }

    fn vmp_apply_dft_to_dft(&self, c: &mut VecZnxDft, a: &VecZnxDft, b: &VmpPMat, tmp_bytes: &mut [u8]) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_to_dft_tmp_bytes(c.cols(), a.cols(), b.rows(), b.cols()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft_to_dft(
                self.ptr,
                c.ptr as *mut vec_znx_dft_t,
                c.cols() as u64,
                a.ptr as *const vec_znx_dft_t,
                a.cols() as u64,
                b.as_ptr() as *const vmp_pmat_t,
                b.rows() as u64,
                b.cols() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vmp_apply_dft_to_dft_add(&self, c: &mut VecZnxDft, a: &VecZnxDft, b: &VmpPMat, tmp_bytes: &mut [u8]) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_to_dft_tmp_bytes(c.cols(), a.cols(), b.rows(), b.cols()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft_to_dft_add(
                self.ptr,
                c.ptr as *mut vec_znx_dft_t,
                c.cols() as u64,
                a.ptr as *const vec_znx_dft_t,
                a.cols() as u64,
                b.as_ptr() as *const vmp_pmat_t,
                b.rows() as u64,
                b.cols() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vmp_apply_dft_to_dft_inplace(&self, b: &mut VecZnxDft, a: &VmpPMat, tmp_bytes: &mut [u8]) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_to_dft_tmp_bytes(b.cols(), b.cols(), a.rows(), a.cols()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft_to_dft(
                self.ptr,
                b.ptr as *mut vec_znx_dft_t,
                b.cols() as u64,
                b.ptr as *mut vec_znx_dft_t,
                b.cols() as u64,
                a.as_ptr() as *const vmp_pmat_t,
                a.rows() as u64,
                a.cols() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Module, Sampling, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, VecZnxOps, VmpPMat, VmpPMatOps, alloc_aligned,
    };
    use sampling::source::Source;

    #[test]
    fn vmp_prepare_row_dft() {
        let module: Module = Module::new(32, crate::BACKEND::FFT64);
        let vpmat_rows: usize = 4;
        let vpmat_cols: usize = 5;
        let log_base2k: usize = 8;
        let mut a: VecZnx = module.new_vec_znx(1, vpmat_cols);
        let mut a_dft: VecZnxDft = module.new_vec_znx_dft(1, vpmat_cols);
        let mut a_big: VecZnxBig = module.new_vec_znx_big(1, vpmat_cols);
        let mut b_big: VecZnxBig = module.new_vec_znx_big(1, vpmat_cols);
        let mut b_dft: VecZnxDft = module.new_vec_znx_dft(1, vpmat_cols);
        let mut vmpmat_0: VmpPMat = module.new_vmp_pmat(1, vpmat_rows, vpmat_cols);
        let mut vmpmat_1: VmpPMat = module.new_vmp_pmat(1, vpmat_rows, vpmat_cols);

        let mut tmp_bytes: Vec<u8> = alloc_aligned(module.vmp_prepare_tmp_bytes(vpmat_rows, vpmat_cols));

        for row_i in 0..vpmat_rows {
            let mut source: Source = Source::new([0u8; 32]);
            module.fill_uniform(log_base2k, &mut a, vpmat_cols, &mut source);
            module.vec_znx_dft(&mut a_dft, &a);
            module.vmp_prepare_row(&mut vmpmat_0, &a.raw(), row_i, &mut tmp_bytes);

            // Checks that prepare(vmp_pmat, a) = prepare_dft(vmp_pmat, a_dft)
            module.vmp_prepare_row_dft(&mut vmpmat_1, &a_dft, row_i);
            assert_eq!(vmpmat_0.raw::<u8>(), vmpmat_1.raw::<u8>());

            // Checks that a_dft = extract_dft(prepare(vmp_pmat, a), b_dft)
            module.vmp_extract_row_dft(&mut b_dft, &vmpmat_0, row_i);
            assert_eq!(a_dft.raw::<u8>(&module), b_dft.raw::<u8>(&module));

            // Checks that a_big = extract(prepare_dft(vmp_pmat, a_dft), b_big)
            module.vmp_extract_row(&mut b_big, &vmpmat_0, row_i);
            module.vec_znx_idft(&mut a_big, &a_dft, &mut tmp_bytes);
            assert_eq!(a_big.raw::<i64>(&module), b_big.raw::<i64>(&module));
        }

        module.free();
    }
}
