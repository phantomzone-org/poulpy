use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::ffi::vmp;
use crate::znx_base::{ZnxInfos, ZnxView, ZnxViewMut};
use crate::{
    Backend, FFT64, MatZnx, MatZnxDft, MatZnxDftPrep, MatZnxDftPrepOwned, MatZnxDftPrepToMut, MatZnxDftPrepToRef, MatZnxDftToRef,
    MatZnxToRef, Module, NTT120, Scratch, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef,
};

pub trait MatZnxDftPrepAlloc<B: Backend> {
    /// Allocates a new [MatZnxDft] with the given number of rows and columns.
    ///
    /// # Arguments
    ///
    /// * `rows`: number of rows (number of [VecZnxDft]).
    /// * `size`: number of size (number of size of each [VecZnxDft]).
    fn new_mat_znx_dft_prep(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxDftPrepOwned<B>;

    fn bytes_of_mat_znx_dft_prep(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;

    fn new_mat_znx_dft_prep_from_bytes(
        &self,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> MatZnxDftPrepOwned<B>;
}

pub trait MatZnxDftPrepScratch {
    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;

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
pub trait MatZnxDftPrepOps<BACKEND: Backend> {
    fn vmp_prepare<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch)
    where
        R: MatZnxDftPrepToMut<BACKEND>,
        A: MatZnxToRef;

    fn vmp_prepare_dft<R, A>(&self, res: &mut R, a: &A)
    where
        R: MatZnxDftPrepToMut<BACKEND>,
        A: MatZnxDftToRef<BACKEND>;

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
        R: VecZnxDftToMut<BACKEND>,
        A: VecZnxDftToRef<BACKEND>,
        B: MatZnxDftPrepToRef<BACKEND>;

    // Same as [MatZnxDftOps::vmp_apply] except result is added on R instead of overwritting R.
    fn vmp_apply_add<R, A, B>(&self, res: &mut R, a: &A, b: &B, scale: usize, scratch: &mut Scratch)
    where
        R: VecZnxDftToMut<BACKEND>,
        A: VecZnxDftToRef<BACKEND>,
        B: MatZnxDftPrepToRef<BACKEND>;
}

impl<B: Backend> MatZnxDftPrepAlloc<B> for Module<B> {
    fn bytes_of_mat_znx_dft_prep(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        MatZnxDftPrepOwned::bytes_of(self, rows, cols_in, cols_out, size)
    }

    fn new_mat_znx_dft_prep(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxDftPrepOwned<B> {
        MatZnxDftPrepOwned::new(self, rows, cols_in, cols_out, size)
    }

    fn new_mat_znx_dft_prep_from_bytes(
        &self,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> MatZnxDftPrepOwned<B> {
        MatZnxDftPrepOwned::new_from_bytes(self, rows, cols_in, cols_out, size, bytes)
    }
}

impl<B: Backend> MatZnxDftPrepScratch for Module<B> {
    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        unsafe { vmp::vmp_prepare_tmp_bytes(self.ptr, (rows * cols_in) as u64, (cols_out * size) as u64) as usize }
    }

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

impl MatZnxDftPrepOps<FFT64> for Module<FFT64> {
    fn vmp_prepare<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch)
    where
        R: MatZnxDftPrepToMut<FFT64>,
        A: MatZnxToRef,
    {
        let mut res: MatZnxDftPrep<&mut [u8], _> = res.to_mut();
        let a: MatZnx<&[u8]> = a.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), self.n());
            assert_eq!(a.n(), self.n());
            assert_eq!(res.cols_in(), a.cols_in());
            assert_eq!(res.rows(), a.rows());
            assert_eq!(res.cols_out(), a.cols_out());
            assert_eq!(res.size(), a.size());
        }

        let (tmp_bytes, _) = scratch.tmp_slice(self.vmp_prepare_tmp_bytes(a.rows(), a.cols_in(), a.cols_out(), a.size()));

        unsafe {
            vmp::vmp_prepare_contiguous(
                self.ptr,
                res.as_mut_ptr() as *mut vmp::vmp_pmat_t,
                a.as_ptr(),
                (a.rows() * a.cols_in()) as u64,
                (a.size() * a.cols_out()) as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vmp_prepare_dft<R, A>(&self, res: &mut R, a: &A)
    where
        R: MatZnxDftPrepToMut<FFT64>,
        A: MatZnxDftToRef<FFT64>,
    {
        let mut res: MatZnxDftPrep<&mut [u8], _> = res.to_mut();
        let a: MatZnxDft<&[u8], _> = a.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), self.n());
            assert_eq!(a.n(), self.n());
            assert_eq!(res.cols_in(), a.cols_in());
            assert_eq!(res.rows(), a.rows());
            assert_eq!(res.cols_out(), a.cols_out());
            assert_eq!(res.size(), a.size());
        }

        unsafe {
            vmp::vmp_prepare_contiguous_dft(
                self.ptr,
                res.as_mut_ptr() as *mut vmp::vmp_pmat_t,
                a.as_ptr(),
                (a.rows() * a.cols_in()) as u64,
                (a.size() * a.cols_out()) as u64,
            );
        }
    }

    fn vmp_apply<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch)
    where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
        B: MatZnxDftPrepToRef<FFT64>,
    {
        let mut res: VecZnxDft<&mut [u8], _> = res.to_mut();
        let a: VecZnxDft<&[u8], _> = a.to_ref();
        let b: MatZnxDftPrep<&[u8], _> = b.to_ref();

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
        B: MatZnxDftPrepToRef<FFT64>,
    {
        let mut res: VecZnxDft<&mut [u8], _> = res.to_mut();
        let a: VecZnxDft<&[u8], _> = a.to_ref();
        let b: MatZnxDftPrep<&[u8], _> = b.to_ref();

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

impl MatZnxDftPrepOps<NTT120> for Module<NTT120> {
    fn vmp_prepare<R, A>(&self, _res: &mut R, _a: &A, _scratch: &mut Scratch)
    where
        R: MatZnxDftPrepToMut<NTT120>,
        A: MatZnxToRef,
    {
        unimplemented!()
    }

    fn vmp_prepare_dft<R, A>(&self, _res: &mut R, _a: &A)
    where
        R: MatZnxDftPrepToMut<NTT120>,
        A: MatZnxDftToRef<NTT120>,
    {
        unimplemented!()
    }

    fn vmp_apply<R, A, B>(&self, _res: &mut R, _a: &A, _b: &B, _scratch: &mut Scratch)
    where
        R: VecZnxDftToMut<NTT120>,
        A: VecZnxDftToRef<NTT120>,
        B: MatZnxDftPrepToRef<NTT120>,
    {
        unimplemented!()
    }

    fn vmp_apply_add<R, A, B>(&self, _res: &mut R, _a: &A, _b: &B, _scale: usize, _scratch: &mut Scratch)
    where
        R: VecZnxDftToMut<NTT120>,
        A: VecZnxDftToRef<NTT120>,
        B: MatZnxDftPrepToRef<NTT120>,
    {
        unimplemented!()
    }
}
