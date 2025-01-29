use crate::bindings::{
    new_vmp_pmat, vmp_apply_dft, vmp_apply_dft_tmp_bytes, vmp_apply_dft_to_dft,
    vmp_apply_dft_to_dft_tmp_bytes, vmp_pmat_t, vmp_prepare_contiguous,
    vmp_prepare_contiguous_tmp_bytes, vmp_prepare_dblptr,
};
use crate::{Module, VecZnx, VecZnxDft};

pub struct VmpPMat(pub *mut vmp_pmat_t, pub usize, pub usize);

impl VmpPMat {
    pub fn rows(&self) -> usize {
        self.1
    }

    pub fn cols(&self) -> usize {
        self.2
    }
}

impl Module {
    pub fn new_vmp_pmat(&self, rows: usize, cols: usize) -> VmpPMat {
        unsafe { VmpPMat(new_vmp_pmat(self.0, rows as u64, cols as u64), rows, cols) }
    }

    pub fn vmp_prepare_contiguous_tmp_bytes(&self, rows: usize, cols: usize) -> usize {
        unsafe { vmp_prepare_contiguous_tmp_bytes(self.0, rows as u64, cols as u64) as usize }
    }

    pub fn vmp_prepare_contiguous(&self, b: &mut VmpPMat, a: &[i64], buf: &mut [u8]) {
        unsafe {
            vmp_prepare_contiguous(
                self.0,
                b.0,
                a.as_ptr(),
                b.1 as u64,
                b.2 as u64,
                buf.as_mut_ptr(),
            );
        }
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
                b.0,
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
                b.0,
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
                a.0,
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

impl<T: Default + Clone> Matrix3D<T> {
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
        let idx: usize = col * (self.n * self.rows) + row * self.n;
        &self.data[idx..idx + self.n]
    }

    pub fn at_mut(&mut self, row: usize, col: usize) -> &mut [T] {
        assert!(row <= self.rows && col <= self.cols);
        let idx: usize = col * (self.n * self.rows) + row * self.n;
        &mut self.data[idx..idx + self.n]
    }
}
