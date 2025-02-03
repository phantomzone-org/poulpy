use crate::ffi::vmp::{
    delete_vmp_pmat, new_vmp_pmat, vmp_apply_dft, vmp_apply_dft_tmp_bytes, vmp_apply_dft_to_dft,
    vmp_apply_dft_to_dft_tmp_bytes, vmp_pmat_t, vmp_prepare_contiguous,
    vmp_prepare_contiguous_tmp_bytes,
};
use crate::{Module, VecZnx, VecZnxDft};
use std::cmp::min;

pub struct VmpPMat {
    pub data: *mut vmp_pmat_t,
    pub rows: usize,
    pub cols: usize,
    pub n: usize,
}

impl VmpPMat {
    pub fn data(&self) -> *mut vmp_pmat_t {
        self.data
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn as_f64(&self) -> &[f64] {
        let ptr: *const f64 = self.data as *const f64;
        let len: usize = (self.rows() * self.cols() * self.n() * 8) / std::mem::size_of::<f64>();
        unsafe { &std::slice::from_raw_parts(ptr, len) }
    }

    pub fn get_addr(&self, row: usize, col: usize, blk: usize) -> &[f64] {
        let nrows: usize = self.rows();
        let ncols: usize = self.cols();
        if col == (ncols - 1) && (ncols & 1 == 1) {
            &self.as_f64()[blk * nrows * ncols * 8 + col * nrows * 8 + row * 8..]
        } else {
            &self.as_f64()[blk * nrows * ncols * 8
                + (col / 2) * (2 * nrows) * 8
                + row * 2 * 8
                + (col % 2) * 8..]
        }
    }

    pub fn at(&self, row: usize, col: usize) -> Vec<f64> {
        //assert!(row <= self.rows && col <= self.cols);

        let mut res: Vec<f64> = vec![f64::default(); self.n];

        if self.n < 8 {
            res.copy_from_slice(
                &self.as_f64()[(row + col * self.rows()) * self.n()
                    ..(row + col * self.rows()) * (self.n() + 1)],
            );
        } else {
            (0..self.n >> 3).for_each(|blk| {
                res[blk * 8..(blk + 1) * 8].copy_from_slice(&self.get_addr(row, col, blk)[..8]);
            });
        }

        res
    }

    pub fn at_mut(&self, row: usize, col: usize) -> &mut [f64] {
        assert!(row <= self.rows && col <= self.cols);
        let idx: usize = col * (self.n / 2 * self.rows) + row * (self.n >> 1);
        let ptr: *mut f64 = self.data as *mut f64;
        let len: usize = (self.rows() * self.cols() * self.n() * 8) / std::mem::size_of::<f64>();
        unsafe { &mut std::slice::from_raw_parts_mut(ptr, len)[idx..idx + self.n] }
    }

    pub fn delete(self) {
        unsafe { delete_vmp_pmat(self.data) };
        drop(self);
    }
}

impl Module {
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

    pub fn vmp_prepare_contiguous_tmp_bytes(&self, rows: usize, cols: usize) -> usize {
        unsafe { vmp_prepare_contiguous_tmp_bytes(self.0, rows as u64, cols as u64) as usize }
    }

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

pub struct Matrix3D<T> {
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
    pub n: usize,
}

impl<T: Default + Clone + std::marker::Copy> Matrix3D<T> {
    pub fn new(rows: usize, cols: usize, n: usize) -> Self {
        let size = rows * cols * n;
        Self {
            data: vec![T::default(); size],
            rows,
            cols,
            n,
        }
    }

    pub fn at(&self, row: usize, col: usize) -> &[T] {
        assert!(row <= self.rows && col <= self.cols);
        let idx: usize = row * (self.n * self.cols) + col * self.n;
        &self.data[idx..idx + self.n]
    }

    pub fn at_mut(&mut self, row: usize, col: usize) -> &mut [T] {
        assert!(row <= self.rows && col <= self.cols);
        let idx: usize = row * (self.n * self.cols) + col * self.n;
        &mut self.data[idx..idx + self.n]
    }

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
