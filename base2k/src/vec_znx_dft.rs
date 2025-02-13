use crate::ffi::vec_znx_big;
use crate::ffi::vec_znx_dft;
use crate::ffi::vec_znx_dft::bytes_of_vec_znx_dft;
use crate::{Module, VecZnx, VecZnxBig};

pub struct VecZnxDft(pub *mut vec_znx_dft::vec_znx_dft_t, pub usize);

impl VecZnxDft {
    /// Returns a new [VecZnxDft] with the provided data as backing array.
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [Module::bytes_of_vec_znx_dft].
    pub fn from_bytes(limbs: usize, data: &mut [u8]) -> VecZnxDft {
        VecZnxDft(data.as_mut_ptr() as *mut vec_znx_dft::vec_znx_dft_t, limbs)
    }

    /// Cast a [VecZnxDft] into a [VecZnxBig].
    /// The returned [VecZnxBig] shares the backing array
    /// with the original [VecZnxDft].
    pub fn as_vec_znx_big(&mut self) -> VecZnxBig {
        VecZnxBig(self.0 as *mut vec_znx_big::vec_znx_bigcoeff_t, self.1)
    }
    pub fn limbs(&self) -> usize {
        self.1
    }
}

impl Module {
    // Allocates a vector Z[X]/(X^N+1) that stores normalized in the DFT space.
    pub fn new_vec_znx_dft(&self, limbs: usize) -> VecZnxDft {
        unsafe { VecZnxDft(vec_znx_dft::new_vec_znx_dft(self.0, limbs as u64), limbs) }
    }

    /// Returns a new [VecZnxDft] with the provided bytes array as backing array.
    ///
    /// # Arguments
    ///
    /// * `limbs`: the number of limbs of the [VecZnxDft].
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_dft].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_dft].
    pub fn new_vec_znx_from_bytes(&self, limbs: usize, bytes: &mut [u8]) -> VecZnxDft {
        assert!(
            bytes.len() >= self.bytes_of_vec_znx_dft(limbs),
            "invalid bytes: bytes.len()={} < bytes_of_vec_znx_dft={}",
            bytes.len(),
            self.bytes_of_vec_znx_dft(limbs)
        );
        VecZnxDft::from_bytes(limbs, bytes)
    }

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnxDft] through [VecZnxDft::from_bytes].
    pub fn bytes_of_vec_znx_dft(&self, limbs: usize) -> usize {
        unsafe { bytes_of_vec_znx_dft(self.0, limbs as u64) as usize }
    }

    // b <- IDFT(a), uses a as scratch space.
    pub fn vec_znx_idft_tmp_a(&self, b: &mut VecZnxBig, a: &mut VecZnxDft, a_limbs: usize) {
        assert!(
            b.limbs() >= a_limbs,
            "invalid c_vector: b_vector.limbs()={} < a_limbs={}",
            b.limbs(),
            a_limbs
        );
        unsafe { vec_znx_dft::vec_znx_idft_tmp_a(self.0, b.0, b.limbs() as u64, a.0, a_limbs as u64) }
    }

    // Returns the size of the scratch space for [vec_znx_idft].
    pub fn vec_znx_idft_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_dft::vec_znx_idft_tmp_bytes(self.0) as usize }
    }

    /// b <- DFT(a)
    ///
    /// # Panics
    /// If b.limbs < a_limbs
    pub fn vec_znx_dft(&self, b: &mut VecZnxDft, a: &VecZnx, a_limbs: usize) {
        assert!(
            b.limbs() >= a_limbs,
            "invalid a_limbs: b.limbs()={} < a_limbs={}",
            b.limbs(),
            a_limbs
        );
        unsafe {
            vec_znx_dft::vec_znx_dft(
                self.0,
                b.0,
                b.limbs() as u64,
                a.as_ptr(),
                a_limbs as u64,
                a.n as u64,
            )
        }
    }

    // b <- IDFT(a), scratch space size obtained with [vec_znx_idft_tmp_bytes].
    pub fn vec_znx_idft(
        &self,
        b: &mut VecZnxBig,
        a: &mut VecZnxDft,
        a_limbs: usize,
        tmp_bytes: &mut [u8],
    ) {
        assert!(
            b.limbs() >= a_limbs,
            "invalid c_vector: b.limbs()={} < a_limbs={}",
            b.limbs(),
            a_limbs
        );
        assert!(
            a.limbs() >= a_limbs,
            "invalid c_vector: a.limbs()={} < a_limbs={}",
            a.limbs(),
            a_limbs
        );
        assert!(
            tmp_bytes.len() <= self.vec_znx_idft_tmp_bytes(),
            "invalid tmp_bytes: tmp_bytes.len()={} < self.vec_znx_idft_tmp_bytes()={}",
            tmp_bytes.len(),
            self.vec_znx_idft_tmp_bytes()
        );
        unsafe {
            vec_znx_dft::vec_znx_idft(
                self.0,
                b.0,
                a.limbs() as u64,
                a.0,
                a_limbs as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }
}
