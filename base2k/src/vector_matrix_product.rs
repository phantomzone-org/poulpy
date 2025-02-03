use crate::ffi::vmp::{
    delete_vmp_pmat, new_vmp_pmat, vmp_apply_dft, vmp_apply_dft_tmp_bytes, vmp_apply_dft_to_dft,
    vmp_apply_dft_to_dft_tmp_bytes, vmp_pmat_t, vmp_prepare_contiguous,
    vmp_prepare_contiguous_tmp_bytes,
};
use crate::{Module, VecZnx, VecZnxDft};
use std::cmp::min;

/// Vector Matrix Product Prepared Matrix: a vector of [VecZnx],
/// stored as a 3D matrix in the DFT domain in a single contiguous array.
pub struct VmpPMat {
    /// The pointer to the C memory.
    pub data: *mut vmp_pmat_t,
    /// The number of [VecZnx].
    pub rows: usize,
    /// The number of limbs in each [VecZnx].      
    pub cols: usize,
    /// The ring degree of each [VecZnx].      
    pub n: usize,
}

impl VmpPMat {

    /// Returns the pointer to the [vmp_pmat_t].
    pub fn data(&self) -> *mut vmp_pmat_t {
        self.data
    }

    /// Returns the number of rows of the [VmpPMat].
    /// The number of rows (i.e. of [VecZnx]) of the [VmpPMat].
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of cols of the [VmpPMat].
    /// The number of cols refers to the number of limbs  
    /// of the prepared [VecZnx].
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the ring dimension of the [VmpPMat].
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns a copy of the backend array at index (i, j) of the [VmpPMat]. 
    /// When using FFT64 as backend, T should be f64.
    /// When using NTT120 as backend, T should be i64.
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

    /// When using FFT64 as backend, T should be f64.
    /// When using NTT120 as backend, T should be i64.
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

    /// Returns a non-mutable reference of T to the entire contiguous array of the [VmpPMat].
    /// When using FFT64 as backend, T should be f64.
    /// When using NTT120 as backend, T should be i64.
    /// The length of the returned array is rows * cols * n.
    pub fn get_backend_array<T>(&self) -> &[T] {
        let ptr: *const T = self.data as *const T;
        let len: usize = (self.rows() * self.cols() * self.n() * 8) / std::mem::size_of::<T>();
        unsafe { &std::slice::from_raw_parts(ptr, len) }
    }

    /// frees the memory and self destructs.
    pub fn delete(self) {
        unsafe { delete_vmp_pmat(self.data) };
        drop(self);
    }
}

impl Module {

    /// Allocates a new [VmpPMat] with the given number of rows and columns.
    pub fn new_vmp_pmat(&self, rows: usize, cols: usize) -> VmpPMat {
        unsafe {
            VmpPMat {
                data: new_vmp_pmat(self.0, rows as u64, cols as u64),
                rows,
                cols,
                n: self.n(),
            }
        }
    }

    /// Returns the number of bytes needed as scratch space for [Self::vmp_prepare_contiguous].
    pub fn vmp_prepare_contiguous_tmp_bytes(&self, rows: usize, cols: usize) -> usize {
        unsafe { vmp_prepare_contiguous_tmp_bytes(self.0, rows as u64, cols as u64) as usize }
    }

    /// Prepares a [VmpPMat] given a contiguous array of [i64].
    /// The helper struct [Matrix3D] can be used to contruct the
    /// appropriate contiguous array.
    /// 
    /// # Example
    /// ```
    /// let mut b_mat: Matrix3D<i64> = Matrix3D::new(rows, cols, n);
    ///
    /// (0..min(rows, cols)).for_each(|i| {
    ///    b_mat.at_mut(i, i)[1] = 1 as i64;
    /// });
    /// let mut vmp_pmat: VmpPMat = module.new_vmp_pmat(rows, cols);
    /// module.vmp_prepare_contiguous(&mut vmp_pmat, &b_mat.data, &mut buf);
    /// ```
    pub fn vmp_prepare_contiguous(&self, b: &mut VmpPMat, a: &[i64], buf: &mut [u8]) {
        unsafe {
            vmp_prepare_contiguous(
                self.0,
                b.data(),
                a.as_ptr(),
                b.rows() as u64,
                b.cols() as u64,
                buf.as_mut_ptr(),
            );
        }
    }

    pub fn vmp_prepare_dblptr(&self, b: &mut VmpPMat, a: &Vec<VecZnx>, buf: &mut [u8]) {
        let rows: usize = b.rows();
        let cols: usize = b.cols();

        let mut mat: Matrix3D<i64> = Matrix3D::<i64>::new(rows, cols, self.n());

        (0..min(rows, a.len())).for_each(|i| {
            mat.set_row(i, &a[i].data);
        });

        self.vmp_prepare_contiguous(b, &mat.data, buf);

        /*
        NOT IMPLEMENTED IN SPQLIOS
        let mut ptrs: Vec<*const i64> = a.iter().map(|v| v.data.as_ptr()).collect();
        unsafe {
            vmp_prepare_dblptr(
                self.0,
                b.data(),
                ptrs.as_mut_ptr(),
                b.rows() as u64,
                b.cols() as u64,
                buf.as_mut_ptr(),
            );
        }
        */
    }

    pub fn vmp_apply_dft_tmp_bytes(
        &self,
        c_limbs: usize,
        a_limbs: usize,
        rows: usize,
        cols: usize,
    ) -> usize {
        unsafe {
            vmp_apply_dft_tmp_bytes(
                self.0,
                c_limbs as u64,
                a_limbs as u64,
                rows as u64,
                cols as u64,
            ) as usize
        }
    }

    pub fn vmp_apply_dft(&self, c: &mut VecZnxDft, a: &VecZnx, b: &VmpPMat, buf: &mut [u8]) {
        unsafe {
            vmp_apply_dft(
                self.0,
                c.0,
                c.limbs() as u64,
                a.as_ptr(),
                a.limbs() as u64,
                a.n() as u64,
                b.data(),
                b.rows() as u64,
                b.cols() as u64,
                buf.as_mut_ptr(),
            )
        }
    }

    pub fn vmp_apply_dft_to_dft_tmp_bytes(
        &self,
        c_limbs: usize,
        a_limbs: usize,
        rows: usize,
        cols: usize,
    ) -> usize {
        unsafe {
            vmp_apply_dft_to_dft_tmp_bytes(
                self.0,
                c_limbs as u64,
                a_limbs as u64,
                rows as u64,
                cols as u64,
            ) as usize
        }
    }

    pub fn vmp_apply_dft_to_dft(
        &self,
        c: &mut VecZnxDft,
        a: &VecZnxDft,
        b: &VmpPMat,
        buf: &mut [u8],
    ) {
        unsafe {
            vmp_apply_dft_to_dft(
                self.0,
                c.0,
                c.limbs() as u64,
                a.0,
                a.limbs() as u64,
                b.data(),
                b.rows() as u64,
                b.cols() as u64,
                buf.as_mut_ptr(),
            )
        }
    }

    pub fn vmp_apply_dft_to_dft_inplace(&self, b: &mut VecZnxDft, a: &VmpPMat, buf: &mut [u8]) {
        unsafe {
            vmp_apply_dft_to_dft(
                self.0,
                b.0,
                b.limbs() as u64,
                b.0,
                b.limbs() as u64,
                a.data(),
                a.rows() as u64,
                a.cols() as u64,
                buf.as_mut_ptr(),
            )
        }
    }
}

/// A helper struture that stores a 3D matrix as a contiguous array.
/// To be passed to [Module::vmp_prepare_contiguous].
///
/// rows: index of the i-th base2K power.
/// cols: index of the j-th limb of the i-th row.
/// n   : polynomial degree.
///
/// A [Matrix3D] can be seen as a vector of [VecZnx].
pub struct Matrix3D<T> {
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
    pub n: usize,
}

impl<T: Default + Clone + std::marker::Copy> Matrix3D<T> {
    /// Allocates a new [Matrix3D] with the respective dimensions.
    ///
    /// # Example
    /// ```
    /// let rows = 5; // #decomp
    /// let cols = 5; // #limbs
    /// let n = 1024; // #coeffs
    ///
    /// let mut mat = Matrix3D::<i64>::new(rows, cols, n);
    /// ```
    pub fn new(rows: usize, cols: usize, n: usize) -> Self {
        let size = rows * cols * n;
        Self {
            data: vec![T::default(); size],
            rows,
            cols,
            n,
        }
    }

    /// Returns a non-mutable reference to the entry (row, col) of the [Matrix3D].
    /// The returned array is of size n.
    ///
    /// # Example
    /// ```
    /// let rows = 5; // #decomp
    /// let cols = 5; // #limbs
    /// let n = 1024; // #coeffs
    ///
    /// let mut mat = Matrix3D::<i64>::new(rows, cols, n);
    ///
    /// let elem: &[i64] = mat.at(5, 5); // size n
    /// ```
    pub fn at(&self, row: usize, col: usize) -> &[T] {
        assert!(row <= self.rows && col <= self.cols);
        let idx: usize = row * (self.n * self.cols) + col * self.n;
        &self.data[idx..idx + self.n]
    }

    /// Returns a mutable reference of the array at the (row, col) entry of the [Matrix3D].
    /// The returned array is of size n.
    ///
    /// # Example
    /// ```
    /// let rows = 5; // #decomp
    /// let cols = 5; // #limbs
    /// let n = 1024; // #coeffs
    ///
    /// let mut mat = Matrix3D::<i64>::new(rows, cols, n);
    ///
    /// let elem: &mut [i64] = mat.at_mut(5, 5); // size n
    /// ```
    pub fn at_mut(&mut self, row: usize, col: usize) -> &mut [T] {
        assert!(row <= self.rows && col <= self.cols);
        let idx: usize = row * (self.n * self.cols) + col * self.n;
        &mut self.data[idx..idx + self.n]
    }

    /// Sets the entry \[row\] of the [Matrix3D].
    /// Typicall this is used to assign a [VecZnx] to the i-th row
    /// of the [Matrix3D].
    ///
    /// # Example
    /// ```
    /// let rows = 5; // #decomp
    /// let cols = 5; // #limbs
    /// let n = 1024; // #coeffs
    ///
    /// let mut mat = Matrix3D::<i64>::new(rows, cols, n);
    ///
    /// let a: Vec<i64> = VecZnx::new(n, cols);
    ///
    /// mat.set_row(1, &a.data);
    /// ```
    pub fn set_row(&mut self, row: usize, a: &[T]) {
        assert!(
            row < self.rows,
            "invalid argument row: row={} > self.rows={}",
            row,
            self.rows
        );
        let idx: usize = row * (self.n * self.cols);
        let size: usize = min(a.len(), self.cols * self.n);
        self.data[idx..idx + size].copy_from_slice(&a[..size]);
    }
}
