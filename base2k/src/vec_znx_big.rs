use crate::ffi::vec_znx_big;
use crate::ffi::vec_znx_dft;
use crate::{is_aligned, Infos, Module, VecZnxApi, VecZnxDft};

pub struct VecZnxBig(pub *mut vec_znx_big::vec_znx_bigcoeff_t, pub usize);

impl VecZnxBig {
    /// Returns a new [VecZnxBig] with the provided data as backing array.
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [Module::bytes_of_vec_znx_big].
    pub fn from_bytes(cols: usize, data: &mut [u8]) -> VecZnxBig {
        debug_assert!(is_aligned(data.as_ptr()));
        VecZnxBig(
            data.as_mut_ptr() as *mut vec_znx_big::vec_znx_bigcoeff_t,
            cols,
        )
    }

    pub fn as_vec_znx_dft(&mut self) -> VecZnxDft {
        VecZnxDft(self.0 as *mut vec_znx_dft::vec_znx_dft_t, self.1)
    }
    pub fn cols(&self) -> usize {
        self.1
    }
}

pub trait VecZnxBigOps {
    /// Allocates a vector Z[X]/(X^N+1) that stores not normalized values.
    fn new_vec_znx_big(&self, cols: usize) -> VecZnxBig;

    /// Returns a new [VecZnxBig] with the provided bytes array as backing array.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of cols of the [VecZnxBig].
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_big].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_big].
    fn new_vec_znx_big_from_bytes(&self, cols: usize, bytes: &mut [u8]) -> VecZnxBig;

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnxBig] through [VecZnxBig::from_bytes].
    fn bytes_of_vec_znx_big(&self, cols: usize) -> usize;

    /// b <- b - a
    fn vec_znx_big_sub_small_a_inplace<T: VecZnxApi + Infos>(&self, b: &mut VecZnxBig, a: &T);

    /// c <- b - a
    fn vec_znx_big_sub_small_a<T: VecZnxApi + Infos>(
        &self,
        c: &mut VecZnxBig,
        a: &T,
        b: &VecZnxBig,
    );

    /// c <- b + a
    fn vec_znx_big_add_small<T: VecZnxApi + Infos>(&self, c: &mut VecZnxBig, a: &T, b: &VecZnxBig);

    /// b <- b + a
    fn vec_znx_big_add_small_inplace<T: VecZnxApi + Infos>(&self, b: &mut VecZnxBig, a: &T);

    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize;

    /// b <- normalize(a)
    fn vec_znx_big_normalize<T: VecZnxApi + Infos>(
        &self,
        log_base2k: usize,
        b: &mut T,
        a: &VecZnxBig,
        tmp_bytes: &mut [u8],
    );

    fn vec_znx_big_range_normalize_base2k_tmp_bytes(&self) -> usize;

    fn vec_znx_big_range_normalize_base2k<T: VecZnxApi + Infos>(
        &self,
        log_base2k: usize,
        res: &mut T,
        a: &VecZnxBig,
        a_range_begin: usize,
        a_range_xend: usize,
        a_range_step: usize,
        tmp_bytes: &mut [u8],
    );

    fn vec_znx_big_automorphism(&self, gal_el: i64, b: &mut VecZnxBig, a: &VecZnxBig);

    fn vec_znx_big_automorphism_inplace(&self, gal_el: i64, a: &mut VecZnxBig);
}

impl VecZnxBigOps for Module {
    fn new_vec_znx_big(&self, cols: usize) -> VecZnxBig {
        unsafe { VecZnxBig(vec_znx_big::new_vec_znx_big(self.0, cols as u64), cols) }
    }

    fn new_vec_znx_big_from_bytes(&self, cols: usize, bytes: &mut [u8]) -> VecZnxBig {
        debug_assert!(
            bytes.len() >= <Module as VecZnxBigOps>::bytes_of_vec_znx_big(self, cols),
            "invalid bytes: bytes.len()={} < bytes_of_vec_znx_dft={}",
            bytes.len(),
            <Module as VecZnxBigOps>::bytes_of_vec_znx_big(self, cols)
        );
        debug_assert!(is_aligned(bytes.as_ptr()));
        VecZnxBig::from_bytes(cols, bytes)
    }

    fn bytes_of_vec_znx_big(&self, cols: usize) -> usize {
        unsafe { vec_znx_big::bytes_of_vec_znx_big(self.0, cols as u64) as usize }
    }

    fn vec_znx_big_sub_small_a_inplace<T: VecZnxApi + Infos>(&self, b: &mut VecZnxBig, a: &T) {
        unsafe {
            vec_znx_big::vec_znx_big_sub_small_a(
                self.0,
                b.0,
                b.cols() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
                b.0,
                b.cols() as u64,
            )
        }
    }

    fn vec_znx_big_sub_small_a<T: VecZnxApi + Infos>(
        &self,
        c: &mut VecZnxBig,
        a: &T,
        b: &VecZnxBig,
    ) {
        unsafe {
            vec_znx_big::vec_znx_big_sub_small_a(
                self.0,
                c.0,
                c.cols() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
                b.0,
                b.cols() as u64,
            )
        }
    }

    fn vec_znx_big_add_small<T: VecZnxApi + Infos>(&self, c: &mut VecZnxBig, a: &T, b: &VecZnxBig) {
        unsafe {
            vec_znx_big::vec_znx_big_add_small(
                self.0,
                c.0,
                c.cols() as u64,
                b.0,
                b.cols() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
            )
        }
    }

    fn vec_znx_big_add_small_inplace<T: VecZnxApi + Infos>(&self, b: &mut VecZnxBig, a: &T) {
        unsafe {
            vec_znx_big::vec_znx_big_add_small(
                self.0,
                b.0,
                b.cols() as u64,
                b.0,
                b.cols() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
            )
        }
    }

    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_big::vec_znx_big_normalize_base2k_tmp_bytes(self.0) as usize }
    }

    fn vec_znx_big_normalize<T: VecZnxApi + Infos>(
        &self,
        log_base2k: usize,
        b: &mut T,
        a: &VecZnxBig,
        tmp_bytes: &mut [u8],
    ) {
        assert!(
            tmp_bytes.len() >= <Module as VecZnxBigOps>::vec_znx_big_normalize_tmp_bytes(self),
            "invalid tmp_bytes: tmp_bytes.len()={} <= self.vec_znx_big_normalize_tmp_bytes()={}",
            tmp_bytes.len(),
            <Module as VecZnxBigOps>::vec_znx_big_normalize_tmp_bytes(self)
        );
        debug_assert!(is_aligned(tmp_bytes.as_ptr()));
        unsafe {
            vec_znx_big::vec_znx_big_normalize_base2k(
                self.0,
                log_base2k as u64,
                b.as_mut_ptr(),
                b.cols() as u64,
                b.n() as u64,
                a.0,
                a.cols() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vec_znx_big_range_normalize_base2k_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_big::vec_znx_big_range_normalize_base2k_tmp_bytes(self.0) as usize }
    }

    fn vec_znx_big_range_normalize_base2k<T: VecZnxApi + Infos>(
        &self,
        log_base2k: usize,
        res: &mut T,
        a: &VecZnxBig,
        a_range_begin: usize,
        a_range_xend: usize,
        a_range_step: usize,
        tmp_bytes: &mut [u8],
    ) {
        assert!(
            tmp_bytes.len() >= <Module as VecZnxBigOps>::vec_znx_big_range_normalize_base2k_tmp_bytes(self),
            "invalid tmp_bytes: tmp_bytes.len()={} <= self.vec_znx_big_range_normalize_base2k_tmp_bytes()={}",
            tmp_bytes.len(),
            <Module as VecZnxBigOps>::vec_znx_big_range_normalize_base2k_tmp_bytes(self)
        );
        debug_assert!(is_aligned(tmp_bytes.as_ptr()));
        unsafe {
            vec_znx_big::vec_znx_big_range_normalize_base2k(
                self.0,
                log_base2k as u64,
                res.as_mut_ptr(),
                res.cols() as u64,
                res.n() as u64,
                a.0,
                a_range_begin as u64,
                a_range_xend as u64,
                a_range_step as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vec_znx_big_automorphism(&self, gal_el: i64, b: &mut VecZnxBig, a: &VecZnxBig) {
        unsafe {
            vec_znx_big::vec_znx_big_automorphism(
                self.0,
                gal_el,
                b.0,
                b.cols() as u64,
                a.0,
                a.cols() as u64,
            );
        }
    }

    fn vec_znx_big_automorphism_inplace(&self, gal_el: i64, a: &mut VecZnxBig) {
        unsafe {
            vec_znx_big::vec_znx_big_automorphism(
                self.0,
                gal_el,
                a.0,
                a.cols() as u64,
                a.0,
                a.cols() as u64,
            );
        }
    }
}
