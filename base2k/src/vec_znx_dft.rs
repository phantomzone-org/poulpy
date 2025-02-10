use crate::ffi::vec_znx_big;
use crate::ffi::vec_znx_dft;
use crate::ffi::vec_znx_dft::bytes_of_vec_znx_dft;
use crate::{Module, VecZnxBig};

pub struct VecZnxDft(pub *mut vec_znx_dft::vec_znx_dft_t, pub usize);

impl VecZnxDft {
    /// Casts a contiguous array of [u8] into as a [VecZnxDft].
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [Module::bytes_of_vec_znx_dft].
    pub fn from_bytes(&self, limbs: usize, data: &mut [u8]) -> VecZnxDft {
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
        unsafe { vec_znx_dft::vec_znx_idft_tmp_a(self.0, b.0, a_limbs as u64, a.0, a_limbs as u64) }
    }

    // Returns the size of the scratch space for [vec_znx_idft].
    pub fn vec_znx_idft_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_dft::vec_znx_idft_tmp_bytes(self.0) as usize }
    }

    // b <- IDFT(a), scratch space size obtained with [vec_znx_idft_tmp_bytes].
    pub fn vec_znx_idft(
        &self,
        b_vector: &mut VecZnxBig,
        a_vector: &mut VecZnxDft,
        a_limbs: usize,
        tmp_bytes: &mut [u8],
    ) {
        assert!(
            b_vector.limbs() >= a_limbs,
            "invalid c_vector: b_vector.limbs()={} < a_limbs={}",
            b_vector.limbs(),
            a_limbs
        );
        assert!(
            a_vector.limbs() >= a_limbs,
            "invalid c_vector: c_vector.limbs()={} < a_limbs={}",
            a_vector.limbs(),
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
                b_vector.0,
                a_limbs as u64,
                a_vector.0,
                a_limbs as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }
}
