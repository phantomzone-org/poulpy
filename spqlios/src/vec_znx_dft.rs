use crate::bindings::{new_vec_znx_dft, vec_znx_idft, vec_znx_idft_tmp_a, vec_znx_idft_tmp_bytes};
use crate::module::{Module, VECZNXBIG, VECZNXDFT};

impl Module {
    // Allocates a vector Z[X]/(X^N+1) that stores normalized in the DFT space.
    pub fn new_vec_znx_dft(&self, limbs: usize) -> VECZNXDFT {
        unsafe { VECZNXDFT(new_vec_znx_dft(self.0, limbs as u64), limbs) }
    }

    // b <- IDFT(a), uses a as scratch space.
    pub fn vec_znx_idft_tmp_a(&self, b: &mut VECZNXBIG, a: &mut VECZNXDFT, a_limbs: usize) {
        assert!(
            b.limbs() >= a_limbs,
            "invalid c_vector: b_vector.limbs()={} < a_limbs={}",
            b.limbs(),
            a_limbs
        );
        unsafe { vec_znx_idft_tmp_a(self.0, b.0, a_limbs as u64, a.0, a_limbs as u64) }
    }

    // Returns the size of the scratch space for [vec_znx_idft].
    pub fn vec_znx_idft_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_idft_tmp_bytes(self.0) as usize }
    }

    // b <- IDFT(a), scratch space size obtained with [vec_znx_idft_tmp_bytes].
    pub fn vec_znx_idft(
        &self,
        b_vector: &mut VECZNXBIG,
        a_vector: &mut VECZNXDFT,
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
            vec_znx_idft(
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
