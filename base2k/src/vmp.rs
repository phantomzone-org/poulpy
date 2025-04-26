use crate::ffi::vec_znx_big::vec_znx_big_t;
use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::ffi::vmp::{self, vmp_pmat_t};
use crate::{Backend, FFT64, Infos, Module, VecZnx, VecZnxBig, VecZnxDft, VecZnxLayout, alloc_aligned, assert_alignement};
use std::marker::PhantomData;

/// Vector Matrix Product Prepared Matrix: a vector of [VecZnx],
/// stored as a 3D matrix in the DFT domain in a single contiguous array.
/// Each col of the [VmpPMat] can be seen as a collection of [VecZnxDft].
///
/// [VmpPMat] is used to permform a vector matrix product between a [VecZnx]/[VecZnxDft] and a [VmpPMat].
/// See the trait [VmpPMatOps] for additional information.
pub struct VmpPMat<B: Backend> {
    /// Raw data, is empty if borrowing scratch space.
    data: Vec<u8>,
    /// Pointer to data. Can point to scratch space.
    ptr: *mut u8,
    /// The ring degree of each polynomial.
    n: usize,
    /// Number of rows
    rows: usize,
    /// Number of cols
    cols: usize,
    /// The number of small polynomials
    limbs: usize,
    _marker: PhantomData<B>,
}

impl<B: Backend> Infos for VmpPMat<B> {
    fn n(&self) -> usize {
        self.n
    }

    fn log_n(&self) -> usize {
        (usize::BITS - (self.n() - 1).leading_zeros()) as _
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn limbs(&self) -> usize {
        self.limbs
    }

    fn poly_count(&self) -> usize {
        self.rows * self.cols * self.limbs
    }
}

impl VmpPMat<FFT64> {
    fn new(module: &Module<FFT64>, rows: usize, cols: usize, limbs: usize) -> VmpPMat<FFT64> {
        let mut data: Vec<u8> = alloc_aligned::<u8>(module.bytes_of_vmp_pmat(rows, cols, limbs));
        let ptr: *mut u8 = data.as_mut_ptr();
        VmpPMat::<FFT64> {
            data: data,
            ptr: ptr,
            n: module.n(),
            rows: rows,
            cols: cols,
            limbs: limbs,
            _marker: PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn borrowed(&self) -> bool {
        self.data.len() == 0
    }

    /// Returns a non-mutable reference to the entire contiguous array of the [VmpPMat].
    pub fn raw(&self) -> &[f64] {
        let ptr: *const f64 = self.ptr as *const f64;
        let size: usize = self.n() * self.poly_count();
        unsafe { &std::slice::from_raw_parts(ptr, size) }
    }

    /// Returns a mutable reference of to the entire contiguous array of the [VmpPMat].
    pub fn raw_mut(&self) -> &mut [f64] {
        let ptr: *mut f64 = self.ptr as *mut f64;
        let size: usize = self.n() * self.poly_count();
        unsafe { std::slice::from_raw_parts_mut(ptr, size) }
    }

    /// Returns a copy of the backend array at index (i, j) of the [VmpPMat].
    ///
    /// # Arguments
    ///
    /// * `row`: row index (i).
    /// * `col`: col index (j).
    pub fn at(&self, row: usize, col: usize) -> Vec<f64> {
        let mut res: Vec<f64> = alloc_aligned(self.n);

        if self.n < 8 {
            res.copy_from_slice(&self.raw()[(row + col * self.rows()) * self.n()..(row + col * self.rows()) * (self.n() + 1)]);
        } else {
            (0..self.n >> 3).for_each(|blk| {
                res[blk * 8..(blk + 1) * 8].copy_from_slice(&self.at_block(row, col, blk)[..8]);
            });
        }

        res
    }

    fn at_block(&self, row: usize, col: usize, blk: usize) -> &[f64] {
        let nrows: usize = self.rows();
        let nsize: usize = self.limbs();
        if col == (nsize - 1) && (nsize & 1 == 1) {
            &self.raw()[blk * nrows * nsize * 8 + col * nrows * 8 + row * 8..]
        } else {
            &self.raw()[blk * nrows * nsize * 8 + (col / 2) * (2 * nrows) * 8 + row * 2 * 8 + (col % 2) * 8..]
        }
    }
}

/// This trait implements methods for vector matrix product,
/// that is, multiplying a [VecZnx] with a [VmpPMat].
pub trait VmpPMatOps<B: Backend> {
    fn bytes_of_vmp_pmat(&self, rows: usize, cols: usize, limbs: usize) -> usize;

    /// Allocates a new [VmpPMat] with the given number of rows and columns.
    ///
    /// # Arguments
    ///
    /// * `rows`: number of rows (number of [VecZnxDft]).
    /// * `size`: number of size (number of size of each [VecZnxDft]).
    fn new_vmp_pmat(&self, rows: usize, cols: usize, limbs: usize) -> VmpPMat<B>;

    /// Returns the number of bytes needed as scratch space for [VmpPMatOps::vmp_prepare_contiguous].
    ///
    /// # Arguments
    ///
    /// * `rows`: number of rows of the [VmpPMat] used in [VmpPMatOps::vmp_prepare_contiguous].
    /// * `size`: number of size of the [VmpPMat] used in [VmpPMatOps::vmp_prepare_contiguous].
    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols: usize, size: usize) -> usize;

    /// Prepares a [VmpPMat] from a contiguous array of [i64].
    /// The helper struct [Matrix3D] can be used to contruct and populate
    /// the appropriate contiguous array.
    ///
    /// # Arguments
    ///
    /// * `b`: [VmpPMat] on which the values are encoded.
    /// * `a`: the contiguous array of [i64] of the 3D matrix to encode on the [VmpPMat].
    /// * `buf`: scratch space, the size of buf can be obtained with [VmpPMatOps::vmp_prepare_tmp_bytes].
    fn vmp_prepare_contiguous(&self, b: &mut VmpPMat<B>, a: &[i64], buf: &mut [u8]);

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
    fn vmp_prepare_row(&self, b: &mut VmpPMat<B>, a: &[i64], row_i: usize, tmp_bytes: &mut [u8]);

    /// Extracts the ith-row of [VmpPMat] into a [VecZnxBig].
    ///
    /// # Arguments
    ///
    /// * `b`: the [VecZnxBig] to on which to extract the row of the [VmpPMat].
    /// * `a`: [VmpPMat] on which the values are encoded.
    /// * `row_i`: the index of the row to extract.
    fn vmp_extract_row(&self, b: &mut VecZnxBig<B>, a: &VmpPMat<B>, row_i: usize);

    /// Prepares the ith-row of [VmpPMat] from a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `b`: [VmpPMat] on which the values are encoded.
    /// * `a`: the [VecZnxDft] to encode on the [VmpPMat].
    /// * `row_i`: the index of the row to prepare.
    ///
    /// The size of buf can be obtained with [VmpPMatOps::vmp_prepare_tmp_bytes].
    fn vmp_prepare_row_dft(&self, b: &mut VmpPMat<B>, a: &VecZnxDft<B>, row_i: usize);

    /// Extracts the ith-row of [VmpPMat] into a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `b`: the [VecZnxDft] to on which to extract the row of the [VmpPMat].
    /// * `a`: [VmpPMat] on which the values are encoded.
    /// * `row_i`: the index of the row to extract.
    fn vmp_extract_row_dft(&self, b: &mut VecZnxDft<B>, a: &VmpPMat<B>, row_i: usize);

    /// Returns the size of the stratch space necessary for [VmpPMatOps::vmp_apply_dft].
    ///
    /// # Arguments
    ///
    /// * `c_size`: number of size of the output [VecZnxDft].
    /// * `a_size`: number of size of the input [VecZnx].
    /// * `rows`: number of rows of the input [VmpPMat].
    /// * `size`: number of size of the input [VmpPMat].
    fn vmp_apply_dft_tmp_bytes(&self, c_size: usize, a_size: usize, rows: usize, size: usize) -> usize;

    /// Applies the vector matrix product [VecZnxDft] x [VmpPMat].
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [VmpPMat].
    ///
    /// As such, given an input [VecZnx] of `i` size and a [VmpPMat] of `i` rows and
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
    /// * `b`: the right operand [VmpPMat] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [VmpPMatOps::vmp_apply_dft_tmp_bytes].
    fn vmp_apply_dft(&self, c: &mut VecZnxDft<B>, a: &VecZnx, b: &VmpPMat<B>, buf: &mut [u8]);

    /// Applies the vector matrix product [VecZnxDft] x [VmpPMat] and adds on the receiver.
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [VmpPMat].
    ///
    /// As such, given an input [VecZnx] of `i` size and a [VmpPMat] of `i` rows and
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
    /// * `b`: the right operand [VmpPMat] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [VmpPMatOps::vmp_apply_dft_tmp_bytes].
    fn vmp_apply_dft_add(&self, c: &mut VecZnxDft<B>, a: &VecZnx, b: &VmpPMat<B>, buf: &mut [u8]);

    /// Returns the size of the stratch space necessary for [VmpPMatOps::vmp_apply_dft_to_dft].
    ///
    /// # Arguments
    ///
    /// * `c_size`: number of size of the output [VecZnxDft].
    /// * `a_size`: number of size of the input [VecZnxDft].
    /// * `rows`: number of rows of the input [VmpPMat].
    /// * `size`: number of size of the input [VmpPMat].
    fn vmp_apply_dft_to_dft_tmp_bytes(&self, c_size: usize, a_size: usize, rows: usize, size: usize) -> usize;

    /// Applies the vector matrix product [VecZnxDft] x [VmpPMat].
    /// The size of `buf` is given by [VmpPMatOps::vmp_apply_dft_to_dft_tmp_bytes].
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [VmpPMat].
    ///
    /// As such, given an input [VecZnx] of `i` size and a [VmpPMat] of `i` rows and
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
    /// * `b`: the right operand [VmpPMat] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [VmpPMatOps::vmp_apply_dft_to_dft_tmp_bytes].
    fn vmp_apply_dft_to_dft(&self, c: &mut VecZnxDft<B>, a: &VecZnxDft<B>, b: &VmpPMat<B>, buf: &mut [u8]);

    /// Applies the vector matrix product [VecZnxDft] x [VmpPMat] and adds on top of the receiver instead of overwritting it.
    /// The size of `buf` is given by [VmpPMatOps::vmp_apply_dft_to_dft_tmp_bytes].
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [VmpPMat].
    ///
    /// As such, given an input [VecZnx] of `i` size and a [VmpPMat] of `i` rows and
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
    /// * `b`: the right operand [VmpPMat] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [VmpPMatOps::vmp_apply_dft_to_dft_tmp_bytes].
    fn vmp_apply_dft_to_dft_add(&self, c: &mut VecZnxDft<B>, a: &VecZnxDft<B>, b: &VmpPMat<B>, buf: &mut [u8]);

    /// Applies the vector matrix product [VecZnxDft] x [VmpPMat] in place.
    /// The size of `buf` is given by [VmpPMatOps::vmp_apply_dft_to_dft_tmp_bytes].
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [VmpPMat].
    ///
    /// As such, given an input [VecZnx] of `i` size and a [VmpPMat] of `i` rows and
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
    /// * `a`: the right operand [VmpPMat] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [VmpPMatOps::vmp_apply_dft_to_dft_tmp_bytes].
    fn vmp_apply_dft_to_dft_inplace(&self, b: &mut VecZnxDft<B>, a: &VmpPMat<B>, buf: &mut [u8]);
}

impl VmpPMatOps<FFT64> for Module<FFT64> {
    fn new_vmp_pmat(&self, rows: usize, cols: usize, limbs: usize) -> VmpPMat<FFT64> {
        VmpPMat::<FFT64>::new(self, rows, cols, limbs)
    }

    fn bytes_of_vmp_pmat(&self, rows: usize, cols: usize, limbs: usize) -> usize {
        unsafe { vmp::bytes_of_vmp_pmat(self.ptr, rows as u64, (limbs * cols) as u64) as usize }
    }

    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols: usize, size: usize) -> usize {
        unsafe { vmp::vmp_prepare_tmp_bytes(self.ptr, rows as u64, (size * cols) as u64) as usize }
    }

    fn vmp_prepare_contiguous(&self, b: &mut VmpPMat<FFT64>, a: &[i64], tmp_bytes: &mut [u8]) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.len(), b.n() * b.poly_count());
            assert!(tmp_bytes.len() >= self.vmp_prepare_tmp_bytes(b.rows(), b.cols(), b.limbs()));
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_prepare_contiguous(
                self.ptr,
                b.as_mut_ptr() as *mut vmp_pmat_t,
                a.as_ptr(),
                b.rows() as u64,
                (b.limbs() * b.cols()) as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vmp_prepare_row(&self, b: &mut VmpPMat<FFT64>, a: &[i64], row_i: usize, tmp_bytes: &mut [u8]) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.len(), b.limbs() * self.n() * b.cols());
            assert!(tmp_bytes.len() >= self.vmp_prepare_tmp_bytes(b.rows(), b.cols(), b.limbs()));
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_prepare_row(
                self.ptr,
                b.as_mut_ptr() as *mut vmp_pmat_t,
                a.as_ptr(),
                row_i as u64,
                b.rows() as u64,
                (b.limbs() * b.cols()) as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vmp_extract_row(&self, b: &mut VecZnxBig<FFT64>, a: &VmpPMat<FFT64>, row_i: usize) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), b.n());
            assert_eq!(a.limbs(), b.limbs());
            assert_eq!(a.cols(), b.cols());
        }
        unsafe {
            vmp::vmp_extract_row(
                self.ptr,
                b.ptr as *mut vec_znx_big_t,
                a.as_ptr() as *const vmp_pmat_t,
                row_i as u64,
                a.rows() as u64,
                (a.limbs() * a.cols()) as u64,
            );
        }
    }

    fn vmp_prepare_row_dft(&self, b: &mut VmpPMat<FFT64>, a: &VecZnxDft<FFT64>, row_i: usize) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), b.n());
            assert_eq!(a.limbs(), b.limbs());
        }
        unsafe {
            vmp::vmp_prepare_row_dft(
                self.ptr,
                b.as_mut_ptr() as *mut vmp_pmat_t,
                a.ptr as *const vec_znx_dft_t,
                row_i as u64,
                b.rows() as u64,
                b.limbs() as u64,
            );
        }
    }

    fn vmp_extract_row_dft(&self, b: &mut VecZnxDft<FFT64>, a: &VmpPMat<FFT64>, row_i: usize) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), b.n());
            assert_eq!(a.limbs(), b.limbs());
        }
        unsafe {
            vmp::vmp_extract_row_dft(
                self.ptr,
                b.ptr as *mut vec_znx_dft_t,
                a.as_ptr() as *const vmp_pmat_t,
                row_i as u64,
                a.rows() as u64,
                a.limbs() as u64,
            );
        }
    }

    fn vmp_apply_dft_tmp_bytes(&self, res_size: usize, a_size: usize, gct_rows: usize, gct_size: usize) -> usize {
        unsafe {
            vmp::vmp_apply_dft_tmp_bytes(
                self.ptr,
                res_size as u64,
                a_size as u64,
                gct_rows as u64,
                gct_size as u64,
            ) as usize
        }
    }

    fn vmp_apply_dft(&self, c: &mut VecZnxDft<FFT64>, a: &VecZnx, b: &VmpPMat<FFT64>, tmp_bytes: &mut [u8]) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_tmp_bytes(c.limbs(), a.limbs(), b.rows(), b.limbs()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft(
                self.ptr,
                c.ptr as *mut vec_znx_dft_t,
                c.limbs() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                (a.n() * a.cols()) as u64,
                b.as_ptr() as *const vmp_pmat_t,
                b.rows() as u64,
                b.limbs() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vmp_apply_dft_add(&self, c: &mut VecZnxDft<FFT64>, a: &VecZnx, b: &VmpPMat<FFT64>, tmp_bytes: &mut [u8]) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_tmp_bytes(c.limbs(), a.limbs(), b.rows(), b.limbs()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft_add(
                self.ptr,
                c.ptr as *mut vec_znx_dft_t,
                c.limbs() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                (a.n() * a.limbs()) as u64,
                b.as_ptr() as *const vmp_pmat_t,
                b.rows() as u64,
                b.limbs() as u64,
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

    fn vmp_apply_dft_to_dft(&self, c: &mut VecZnxDft<FFT64>, a: &VecZnxDft<FFT64>, b: &VmpPMat<FFT64>, tmp_bytes: &mut [u8]) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_to_dft_tmp_bytes(c.limbs(), a.limbs(), b.rows(), b.limbs()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft_to_dft(
                self.ptr,
                c.ptr as *mut vec_znx_dft_t,
                c.limbs() as u64,
                a.ptr as *const vec_znx_dft_t,
                a.limbs() as u64,
                b.as_ptr() as *const vmp_pmat_t,
                b.rows() as u64,
                b.limbs() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vmp_apply_dft_to_dft_add(&self, c: &mut VecZnxDft<FFT64>, a: &VecZnxDft<FFT64>, b: &VmpPMat<FFT64>, tmp_bytes: &mut [u8]) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_to_dft_tmp_bytes(c.limbs(), a.limbs(), b.rows(), b.limbs()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft_to_dft_add(
                self.ptr,
                c.ptr as *mut vec_znx_dft_t,
                c.limbs() as u64,
                a.ptr as *const vec_znx_dft_t,
                a.limbs() as u64,
                b.as_ptr() as *const vmp_pmat_t,
                b.rows() as u64,
                b.limbs() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vmp_apply_dft_to_dft_inplace(&self, b: &mut VecZnxDft<FFT64>, a: &VmpPMat<FFT64>, tmp_bytes: &mut [u8]) {
        debug_assert!(tmp_bytes.len() >= self.vmp_apply_dft_to_dft_tmp_bytes(b.limbs(), b.limbs(), a.rows(), a.limbs()));
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vmp::vmp_apply_dft_to_dft(
                self.ptr,
                b.ptr as *mut vec_znx_dft_t,
                b.limbs() as u64,
                b.ptr as *mut vec_znx_dft_t,
                b.limbs() as u64,
                a.as_ptr() as *const vmp_pmat_t,
                a.rows() as u64,
                a.limbs() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        FFT64, Module, Sampling, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, VecZnxLayout, VecZnxOps, VmpPMat,
        VmpPMatOps, alloc_aligned,
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
        let mut vmpmat_0: VmpPMat<FFT64> = module.new_vmp_pmat(vpmat_rows, 1, vpmat_size);
        let mut vmpmat_1: VmpPMat<FFT64> = module.new_vmp_pmat(vpmat_rows, 1, vpmat_size);

        let mut tmp_bytes: Vec<u8> = alloc_aligned(module.vmp_prepare_tmp_bytes(vpmat_rows, 1, vpmat_size));

        for row_i in 0..vpmat_rows {
            let mut source: Source = Source::new([0u8; 32]);
            module.fill_uniform(log_base2k, &mut a, 0, vpmat_size, &mut source);
            module.vec_znx_dft(&mut a_dft, &a);
            module.vmp_prepare_row(&mut vmpmat_0, &a.raw(), row_i, &mut tmp_bytes);

            // Checks that prepare(vmp_pmat, a) = prepare_dft(vmp_pmat, a_dft)
            module.vmp_prepare_row_dft(&mut vmpmat_1, &a_dft, row_i);
            assert_eq!(vmpmat_0.raw(), vmpmat_1.raw());

            // Checks that a_dft = extract_dft(prepare(vmp_pmat, a), b_dft)
            module.vmp_extract_row_dft(&mut b_dft, &vmpmat_0, row_i);
            assert_eq!(a_dft.raw(), b_dft.raw());

            // Checks that a_big = extract(prepare_dft(vmp_pmat, a_dft), b_big)
            module.vmp_extract_row(&mut b_big, &vmpmat_0, row_i);
            module.vec_znx_idft(&mut a_big, &a_dft, &mut tmp_bytes);
            assert_eq!(a_big.raw(), b_big.raw());
        }

        module.free();
    }
}
