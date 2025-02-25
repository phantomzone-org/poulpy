use crate::ffi::vec_znx_big;
use crate::ffi::vec_znx_dft;
use crate::ffi::vec_znx_dft::bytes_of_vec_znx_dft;
use crate::{assert_alignement, Infos, Module, VecZnxApi, VecZnxBig};

pub struct VecZnxDft(pub *mut vec_znx_dft::vec_znx_dft_t, pub usize);

impl VecZnxDft {
    /// Returns a new [VecZnxDft] with the provided data as backing array.
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [Module::bytes_of_vec_znx_dft].
    pub fn from_bytes(cols: usize, tmp_bytes: &mut [u8]) -> VecZnxDft {
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr())
        }
        VecZnxDft(
            tmp_bytes.as_mut_ptr() as *mut vec_znx_dft::vec_znx_dft_t,
            cols,
        )
    }

    /// Cast a [VecZnxDft] into a [VecZnxBig].
    /// The returned [VecZnxBig] shares the backing array
    /// with the original [VecZnxDft].
    pub fn as_vec_znx_big(&mut self) -> VecZnxBig {
        VecZnxBig(self.0 as *mut vec_znx_big::vec_znx_bigcoeff_t, self.1)
    }
    pub fn cols(&self) -> usize {
        self.1
    }
}

pub trait VecZnxDftOps {
    /// Allocates a vector Z[X]/(X^N+1) that stores normalized in the DFT space.
    fn new_vec_znx_dft(&self, cols: usize) -> VecZnxDft;

    /// Returns a new [VecZnxDft] with the provided bytes array as backing array.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of cols of the [VecZnxDft].
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_dft].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_dft].
    fn new_vec_znx_dft_from_bytes(&self, cols: usize, bytes: &mut [u8]) -> VecZnxDft;

    /// Returns a new [VecZnxDft] with the provided bytes array as backing array.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of cols of the [VecZnxDft].
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_dft].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_dft].
    fn bytes_of_vec_znx_dft(&self, cols: usize) -> usize;

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnxDft] through [VecZnxDft::from_bytes].
    fn vec_znx_idft_tmp_bytes(&self) -> usize;

    /// b <- IDFT(a), uses a as scratch space.
    fn vec_znx_idft_tmp_a(&self, b: &mut VecZnxBig, a: &mut VecZnxDft, a_limbs: usize);

    fn vec_znx_idft(
        &self,
        b: &mut VecZnxBig,
        a: &mut VecZnxDft,
        a_limbs: usize,
        tmp_bytes: &mut [u8],
    );

    fn vec_znx_dft<T: VecZnxApi + Infos>(&self, b: &mut VecZnxDft, a: &T, a_limbs: usize);
}

impl VecZnxDftOps for Module {
    fn new_vec_znx_dft(&self, cols: usize) -> VecZnxDft {
        unsafe { VecZnxDft(vec_znx_dft::new_vec_znx_dft(self.0, cols as u64), cols) }
    }

    fn new_vec_znx_dft_from_bytes(&self, cols: usize, tmp_bytes: &mut [u8]) -> VecZnxDft {
        debug_assert!(
            tmp_bytes.len() >= <Module as VecZnxDftOps>::bytes_of_vec_znx_dft(self, cols),
            "invalid bytes: bytes.len()={} < bytes_of_vec_znx_dft={}",
            tmp_bytes.len(),
            <Module as VecZnxDftOps>::bytes_of_vec_znx_dft(self, cols)
        );
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr())
        }
        VecZnxDft::from_bytes(cols, tmp_bytes)
    }

    fn bytes_of_vec_znx_dft(&self, cols: usize) -> usize {
        unsafe { bytes_of_vec_znx_dft(self.0, cols as u64) as usize }
    }

    fn vec_znx_idft_tmp_a(&self, b: &mut VecZnxBig, a: &mut VecZnxDft, a_limbs: usize) {
        debug_assert!(
            b.cols() >= a_limbs,
            "invalid c_vector: b_vector.cols()={} < a_limbs={}",
            b.cols(),
            a_limbs
        );
        unsafe {
            vec_znx_dft::vec_znx_idft_tmp_a(self.0, b.0, b.cols() as u64, a.0, a_limbs as u64)
        }
    }

    fn vec_znx_idft_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_dft::vec_znx_idft_tmp_bytes(self.0) as usize }
    }

    /// b <- DFT(a)
    ///
    /// # Panics
    /// If b.cols < a_cols
    fn vec_znx_dft<T: VecZnxApi + Infos>(&self, b: &mut VecZnxDft, a: &T, a_cols: usize) {
        debug_assert!(
            b.cols() >= a_cols,
            "invalid a_cols: b.cols()={} < a_cols={}",
            b.cols(),
            a_cols
        );
        unsafe {
            vec_znx_dft::vec_znx_dft(
                self.0,
                b.0,
                b.cols() as u64,
                a.as_ptr(),
                a_cols as u64,
                a.n() as u64,
            )
        }
    }

    // b <- IDFT(a), scratch space size obtained with [vec_znx_idft_tmp_bytes].
    fn vec_znx_idft(
        &self,
        b: &mut VecZnxBig,
        a: &mut VecZnxDft,
        a_cols: usize,
        tmp_bytes: &mut [u8],
    ) {
        debug_assert!(
            b.cols() >= a_cols,
            "invalid c_vector: b.cols()={} < a_cols={}",
            b.cols(),
            a_cols
        );
        debug_assert!(
            a.cols() >= a_cols,
            "invalid c_vector: a.cols()={} < a_cols={}",
            a.cols(),
            a_cols
        );
        debug_assert!(
            tmp_bytes.len() <= <Module as VecZnxDftOps>::vec_znx_idft_tmp_bytes(self),
            "invalid tmp_bytes: tmp_bytes.len()={} < self.vec_znx_idft_tmp_bytes()={}",
            tmp_bytes.len(),
            <Module as VecZnxDftOps>::vec_znx_idft_tmp_bytes(self)
        );
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr())
        }
        unsafe {
            vec_znx_dft::vec_znx_idft(
                self.0,
                b.0,
                a.cols() as u64,
                a.0,
                a_cols as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }
}
