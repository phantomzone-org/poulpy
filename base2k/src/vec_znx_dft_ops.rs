use crate::VecZnxDftOwned;
use crate::ffi::vec_znx_big;
use crate::ffi::vec_znx_dft;
use crate::znx_base::ZnxAlloc;
use crate::znx_base::ZnxInfos;
use crate::{FFT64, Module, VecZnx, VecZnxBig, VecZnxDft, ZnxView, ZnxViewMut, ZnxZero, assert_alignement};
use std::cmp::min;

pub trait VecZnxDftAlloc<B> {
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

    // /// Returns a new [VecZnxDft] with the provided bytes array as backing array.
    // ///
    // /// Behavior: the backing array is only borrowed.
    // ///
    // /// # Arguments
    // ///
    // /// * `cols`: the number of cols of the [VecZnxDft].
    // /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_dft].
    // ///
    // /// # Panics
    // /// If `bytes.len()` < [Module::bytes_of_vec_znx_dft].
    // fn new_vec_znx_dft_from_bytes_borrow(&self, cols: usize, size: usize, bytes: &mut [u8]) -> VecZnxDft<B>;

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

pub trait VecZnxDftOps<DataMut, Data, B> {
    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnxDft] through [VecZnxDft::from_bytes].
    fn vec_znx_idft_tmp_bytes(&self) -> usize;

    /// b <- IDFT(a), uses a as scratch space.
    fn vec_znx_idft_tmp_a(&self, res: &mut VecZnxBig<DataMut, B>, res_col: usize, a: &mut VecZnxDft<DataMut, B>, a_cols: usize);

    fn vec_znx_idft(
        &self,
        res: &mut VecZnxBig<DataMut, B>,
        res_col: usize,
        a: &VecZnxDft<Data, B>,
        a_col: usize,
        tmp_bytes: &mut [u8],
    );

    fn vec_znx_dft(&self, res: &mut VecZnxDft<DataMut, B>, res_col: usize, a: &VecZnx<Data>, a_col: usize);
}

impl VecZnxDftAlloc<FFT64> for Module<FFT64> {
    fn new_vec_znx_dft(&self, cols: usize, size: usize) -> VecZnxDftOwned<FFT64> {
        VecZnxDftOwned::new(&self, cols, size)
    }

    fn new_vec_znx_dft_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<FFT64> {
        VecZnxDftOwned::new_from_bytes(self, cols, size, bytes)
    }

    // fn new_vec_znx_dft_from_bytes_borrow(&self, cols: usize, size: usize, bytes: &mut [u8]) -> VecZnxDft<FFT64> {
    //     VecZnxDft::from_bytes_borrow(self, 1, cols, size, bytes)
    // }

    fn bytes_of_vec_znx_dft(&self, cols: usize, size: usize) -> usize {
        VecZnxDft::bytes_of(&self, cols, size)
    }
}

impl<DataMut, Data> VecZnxDftOps<DataMut, Data, FFT64> for Module<FFT64>
where
    DataMut: AsMut<[u8]> + AsRef<[u8]>,
    Data: AsRef<[u8]>,
{
    fn vec_znx_idft_tmp_a(
        &self,
        res: &mut VecZnxBig<DataMut, FFT64>,
        res_col: usize,
        a: &mut VecZnxDft<DataMut, FFT64>,
        a_col: usize,
    ) {
        let min_size: usize = min(res.size(), a.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_znx_idft_tmp_a(
                    self.ptr,
                    res.at_mut_ptr(res_col, j) as *mut vec_znx_big::vec_znx_big_t,
                    1 as u64,
                    a.at_mut_ptr(a_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                    1 as u64,
                )
            });
            (min_size..res.size()).for_each(|j| {
                res.zero_at(res_col, j);
            })
        }
    }

    fn vec_znx_idft_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_dft::vec_znx_idft_tmp_bytes(self.ptr) as usize }
    }

    /// b <- DFT(a)
    ///
    /// # Panics
    /// If b.cols < a_cols
    fn vec_znx_dft(&self, res: &mut VecZnxDft<DataMut, FFT64>, res_col: usize, a: &VecZnx<Data>, a_col: usize) {
        let min_size: usize = min(res.size(), a.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_znx_dft(
                    self.ptr,
                    res.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                    1 as u64,
                    a.at_ptr(a_col, j),
                    1 as u64,
                    a.sl() as u64,
                )
            });
            (min_size..res.size()).for_each(|j| {
                res.zero_at(res_col, j);
            });
        }
    }

    // b <- IDFT(a), scratch space size obtained with [vec_znx_idft_tmp_bytes].
    fn vec_znx_idft(
        &self,
        res: &mut VecZnxBig<DataMut, FFT64>,
        res_col: usize,
        a: &VecZnxDft<Data, FFT64>,
        a_col: usize,
        tmp_bytes: &mut [u8],
    ) {
        #[cfg(debug_assertions)]
        {
            assert!(
                tmp_bytes.len() >= Self::vec_znx_idft_tmp_bytes(self),
                "invalid tmp_bytes: tmp_bytes.len()={} < self.vec_znx_idft_tmp_bytes()={}",
                tmp_bytes.len(),
                Self::vec_znx_idft_tmp_bytes(self)
            );
            assert_alignement(tmp_bytes.as_ptr())
        }

        let min_size: usize = min(res.size(), a.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_znx_idft(
                    self.ptr,
                    res.at_mut_ptr(res_col, j) as *mut vec_znx_big::vec_znx_big_t,
                    1 as u64,
                    a.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1 as u64,
                    tmp_bytes.as_mut_ptr(),
                )
            });
            (min_size..res.size()).for_each(|j| {
                res.zero_at(res_col, j);
            });
        }
    }
}
