use crate::ffi::{vec_znx_big, vec_znx_dft};
use crate::vec_znx_dft::bytes_of_vec_znx_dft;
use crate::znx_base::ZnxInfos;
use crate::{
    Backend, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftOwned, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef,
    ZnxSliceSize,
};
use crate::{FFT64, Module, ZnxView, ZnxViewMut, ZnxZero};
use std::cmp::min;

pub trait VecZnxDftAlloc<B: Backend> {
    /// Allocates a vector Z[X]/(X^N+1) that stores normalized in the DFT space.
    fn new_vec_znx_dft(&self, cols: usize, size: usize) -> VecZnxDftOwned<B>;

    /// Returns a new [VecZnxDft] with the provided bytes array as backing array.
    ///
    /// Behavior: takes ownership of the backing array.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of cols of the [VecZnxDft].
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_dft].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_dft].
    fn new_vec_znx_dft_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<B>;

    /// Returns a new [VecZnxDft] with the provided bytes array as backing array.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of cols of the [VecZnxDft].
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_dft].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_dft].
    fn bytes_of_vec_znx_dft(&self, cols: usize, size: usize) -> usize;
}

pub trait VecZnxDftOps<B: Backend> {
    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnxDft] through [VecZnxDft::from_bytes].
    fn vec_znx_idft_tmp_bytes(&self) -> usize;

    fn vec_znx_dft_copy<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;

    /// b <- IDFT(a), uses a as scratch space.
    fn vec_znx_idft_tmp_a<R, A>(&self, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToMut<B>;

    /// Consumes a to return IDFT(a) in big coeff space.
    fn vec_znx_idft_consume<D>(&self, a: VecZnxDft<D, B>) -> VecZnxBig<D, FFT64>
    where
        VecZnxDft<D, FFT64>: VecZnxDftToMut<FFT64>;

    fn vec_znx_idft<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToRef<B>;

    fn vec_znx_dft<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef;
}

impl<B: Backend> VecZnxDftAlloc<B> for Module<B> {
    fn new_vec_znx_dft(&self, cols: usize, size: usize) -> VecZnxDftOwned<B> {
        VecZnxDftOwned::new(&self, cols, size)
    }

    fn new_vec_znx_dft_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<B> {
        VecZnxDftOwned::new_from_bytes(self, cols, size, bytes)
    }

    fn bytes_of_vec_znx_dft(&self, cols: usize, size: usize) -> usize {
        bytes_of_vec_znx_dft(self, cols, size)
    }
}

impl VecZnxDftOps<FFT64> for Module<FFT64> {
    fn vec_znx_dft_copy<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
    {
        let mut res_mut: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], FFT64> = a.to_ref();

        let min_size: usize = min(res_mut.size(), a_ref.size());

        (0..min_size).for_each(|j| {
            res_mut
                .at_mut(res_col, j)
                .copy_from_slice(a_ref.at(a_col, j));
        });
        (min_size..res_mut.size()).for_each(|j| {
            res_mut.zero_at(res_col, j);
        })
    }

    fn vec_znx_idft_tmp_a<R, A>(&self, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxDftToMut<FFT64>,
    {
        let mut res_mut: VecZnxBig<&mut [u8], FFT64> = res.to_mut();
        let mut a_mut: VecZnxDft<&mut [u8], FFT64> = a.to_mut();

        let min_size: usize = min(res_mut.size(), a_mut.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_znx_idft_tmp_a(
                    self.ptr,
                    res_mut.at_mut_ptr(res_col, j) as *mut vec_znx_big::vec_znx_big_t,
                    1 as u64,
                    a_mut.at_mut_ptr(a_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                    1 as u64,
                )
            });
            (min_size..res_mut.size()).for_each(|j| {
                res_mut.zero_at(res_col, j);
            })
        }
    }

    fn vec_znx_idft_consume<D>(&self, mut a: VecZnxDft<D, FFT64>) -> VecZnxBig<D, FFT64>
    where
        VecZnxDft<D, FFT64>: VecZnxDftToMut<FFT64>,
    {
        let mut a_mut: VecZnxDft<&mut [u8], FFT64> = a.to_mut();

        unsafe {
            // Rev col and rows because ZnxDft.sl() >= ZnxBig.sl()
            (0..a_mut.size()).for_each(|j| {
                (0..a_mut.cols()).for_each(|i| {
                    vec_znx_dft::vec_znx_idft_tmp_a(
                        self.ptr,
                        a_mut.at_mut_ptr(i, j) as *mut vec_znx_big::vec_znx_big_t,
                        1 as u64,
                        a_mut.at_mut_ptr(i, j) as *mut vec_znx_dft::vec_znx_dft_t,
                        1 as u64,
                    )
                });
            });
        }

        a.into_big()
    }

    fn vec_znx_idft_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_dft::vec_znx_idft_tmp_bytes(self.ptr) as usize }
    }

    /// b <- DFT(a)
    ///
    /// # Panics
    /// If b.cols < a_col
    fn vec_znx_dft<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxToRef,
    {
        let mut res_mut: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a_ref: crate::VecZnx<&[u8]> = a.to_ref();

        let min_size: usize = min(res_mut.size(), a_ref.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_znx_dft(
                    self.ptr,
                    res_mut.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                    1 as u64,
                    a_ref.at_ptr(a_col, j),
                    1 as u64,
                    a_ref.sl() as u64,
                )
            });
            (min_size..res_mut.size()).for_each(|j| {
                res_mut.zero_at(res_col, j);
            });
        }
    }

    // b <- IDFT(a), scratch space size obtained with [vec_znx_idft_tmp_bytes].
    fn vec_znx_idft<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
    {
        let mut res_mut: VecZnxBig<&mut [u8], FFT64> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], FFT64> = a.to_ref();

        let (tmp_bytes, _) = scratch.tmp_slice(self.vec_znx_idft_tmp_bytes());

        let min_size: usize = min(res_mut.size(), a_ref.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_znx_idft(
                    self.ptr,
                    res_mut.at_mut_ptr(res_col, j) as *mut vec_znx_big::vec_znx_big_t,
                    1 as u64,
                    a_ref.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1 as u64,
                    tmp_bytes.as_mut_ptr(),
                )
            });
            (min_size..res_mut.size()).for_each(|j| {
                res_mut.zero_at(res_col, j);
            });
        }
    }
}
