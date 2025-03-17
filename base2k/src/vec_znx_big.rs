use crate::ffi::vec_znx_big::{self, vec_znx_bigcoeff_t};
use crate::{alloc_aligned, assert_alignement, Infos, Module, VecZnx, VecZnxDft, MODULETYPE};

pub struct VecZnxBig {
    pub data: Vec<u8>,
    pub ptr: *mut u8,
    pub n: usize,
    pub cols: usize,
    pub backend: MODULETYPE,
}

impl VecZnxBig {
    /// Returns a new [VecZnxBig] with the provided data as backing array.
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [Module::bytes_of_vec_znx_big].
    pub fn from_bytes(module: &Module, cols: usize, bytes: &mut [u8]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert_alignement(bytes.as_ptr())
        };
        unsafe {
            Self {
                data: Vec::from_raw_parts(bytes.as_mut_ptr(), bytes.len(), bytes.len()),
                ptr: bytes.as_mut_ptr(),
                n: module.n(),
                cols: cols,
                backend: module.backend,
            }
        }
    }

    pub fn from_bytes_borrow(module: &Module, cols: usize, bytes: &mut [u8]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert_eq!(bytes.len(), module.bytes_of_vec_znx_big(cols));
            assert_alignement(bytes.as_ptr());
        }
        Self {
            data: Vec::new(),
            ptr: bytes.as_mut_ptr(),
            n: module.n(),
            cols: cols,
            backend: module.backend,
        }
    }

    pub fn as_vec_znx_dft(&mut self) -> VecZnxDft {
        VecZnxDft {
            data: Vec::new(),
            ptr: self.ptr,
            n: self.n,
            cols: self.cols,
            backend: self.backend,
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn backend(&self) -> MODULETYPE {
        self.backend
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
    fn vec_znx_big_sub_small_a_inplace(&self, b: &mut VecZnxBig, a: &VecZnx);

    /// c <- b - a
    fn vec_znx_big_sub_small_a(&self, c: &mut VecZnxBig, a: &VecZnx, b: &VecZnxBig);

    /// c <- b + a
    fn vec_znx_big_add_small(&self, c: &mut VecZnxBig, a: &VecZnx, b: &VecZnxBig);

    /// b <- b + a
    fn vec_znx_big_add_small_inplace(&self, b: &mut VecZnxBig, a: &VecZnx);

    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize;

    /// b <- normalize(a)
    fn vec_znx_big_normalize(
        &self,
        log_base2k: usize,
        b: &mut VecZnx,
        a: &VecZnxBig,
        tmp_bytes: &mut [u8],
    );

    fn vec_znx_big_range_normalize_base2k_tmp_bytes(&self) -> usize;

    fn vec_znx_big_range_normalize_base2k(
        &self,
        log_base2k: usize,
        res: &mut VecZnx,
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
        let mut data: Vec<u8> = alloc_aligned::<u8>(self.bytes_of_vec_znx_big(cols));
        let ptr: *mut u8 = data.as_mut_ptr();
        VecZnxBig {
            data: data,
            ptr: ptr,
            n: self.n(),
            cols: cols,
            backend: self.backend(),
        }
    }

    fn new_vec_znx_big_from_bytes(&self, cols: usize, bytes: &mut [u8]) -> VecZnxBig {
        debug_assert!(
            bytes.len() >= <Module as VecZnxBigOps>::bytes_of_vec_znx_big(self, cols),
            "invalid bytes: bytes.len()={} < bytes_of_vec_znx_dft={}",
            bytes.len(),
            <Module as VecZnxBigOps>::bytes_of_vec_znx_big(self, cols)
        );
        #[cfg(debug_assertions)]
        {
            assert_alignement(bytes.as_ptr())
        }
        VecZnxBig::from_bytes(self, cols, bytes)
    }

    fn bytes_of_vec_znx_big(&self, cols: usize) -> usize {
        unsafe { vec_znx_big::bytes_of_vec_znx_big(self.ptr, cols as u64) as usize }
    }

    fn vec_znx_big_sub_small_a_inplace(&self, b: &mut VecZnxBig, a: &VecZnx) {
        unsafe {
            vec_znx_big::vec_znx_big_sub_small_a(
                self.ptr,
                b.ptr as *mut vec_znx_bigcoeff_t,
                b.cols() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
                b.ptr as *mut vec_znx_bigcoeff_t,
                b.cols() as u64,
            )
        }
    }

    fn vec_znx_big_sub_small_a(&self, c: &mut VecZnxBig, a: &VecZnx, b: &VecZnxBig) {
        unsafe {
            vec_znx_big::vec_znx_big_sub_small_a(
                self.ptr,
                c.ptr as *mut vec_znx_bigcoeff_t,
                c.cols() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
                b.ptr as *mut vec_znx_bigcoeff_t,
                b.cols() as u64,
            )
        }
    }

    fn vec_znx_big_add_small(&self, c: &mut VecZnxBig, a: &VecZnx, b: &VecZnxBig) {
        unsafe {
            vec_znx_big::vec_znx_big_add_small(
                self.ptr,
                c.ptr as *mut vec_znx_bigcoeff_t,
                c.cols() as u64,
                b.ptr as *mut vec_znx_bigcoeff_t,
                b.cols() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
            )
        }
    }

    fn vec_znx_big_add_small_inplace(&self, b: &mut VecZnxBig, a: &VecZnx) {
        unsafe {
            vec_znx_big::vec_znx_big_add_small(
                self.ptr,
                b.ptr as *mut vec_znx_bigcoeff_t,
                b.cols() as u64,
                b.ptr as *mut vec_znx_bigcoeff_t,
                b.cols() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
            )
        }
    }

    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_big::vec_znx_big_normalize_base2k_tmp_bytes(self.ptr) as usize }
    }

    fn vec_znx_big_normalize(
        &self,
        log_base2k: usize,
        b: &mut VecZnx,
        a: &VecZnxBig,
        tmp_bytes: &mut [u8],
    ) {
        debug_assert!(
            tmp_bytes.len() >= <Module as VecZnxBigOps>::vec_znx_big_normalize_tmp_bytes(self),
            "invalid tmp_bytes: tmp_bytes.len()={} <= self.vec_znx_big_normalize_tmp_bytes()={}",
            tmp_bytes.len(),
            <Module as VecZnxBigOps>::vec_znx_big_normalize_tmp_bytes(self)
        );
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr())
        }
        unsafe {
            vec_znx_big::vec_znx_big_normalize_base2k(
                self.ptr,
                log_base2k as u64,
                b.as_mut_ptr(),
                b.cols() as u64,
                b.n() as u64,
                a.ptr as *mut vec_znx_bigcoeff_t,
                a.cols() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vec_znx_big_range_normalize_base2k_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_big::vec_znx_big_range_normalize_base2k_tmp_bytes(self.ptr) as usize }
    }

    fn vec_znx_big_range_normalize_base2k(
        &self,
        log_base2k: usize,
        res: &mut VecZnx,
        a: &VecZnxBig,
        a_range_begin: usize,
        a_range_xend: usize,
        a_range_step: usize,
        tmp_bytes: &mut [u8],
    ) {
        debug_assert!(
            tmp_bytes.len() >= <Module as VecZnxBigOps>::vec_znx_big_range_normalize_base2k_tmp_bytes(self),
            "invalid tmp_bytes: tmp_bytes.len()={} <= self.vec_znx_big_range_normalize_base2k_tmp_bytes()={}",
            tmp_bytes.len(),
            <Module as VecZnxBigOps>::vec_znx_big_range_normalize_base2k_tmp_bytes(self)
        );
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr())
        }
        unsafe {
            vec_znx_big::vec_znx_big_range_normalize_base2k(
                self.ptr,
                log_base2k as u64,
                res.as_mut_ptr(),
                res.cols() as u64,
                res.n() as u64,
                a.ptr as *mut vec_znx_bigcoeff_t,
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
                self.ptr,
                gal_el,
                b.ptr as *mut vec_znx_bigcoeff_t,
                b.cols() as u64,
                a.ptr as *mut vec_znx_bigcoeff_t,
                a.cols() as u64,
            );
        }
    }

    fn vec_znx_big_automorphism_inplace(&self, gal_el: i64, a: &mut VecZnxBig) {
        unsafe {
            vec_znx_big::vec_znx_big_automorphism(
                self.ptr,
                gal_el,
                a.ptr as *mut vec_znx_bigcoeff_t,
                a.cols() as u64,
                a.ptr as *mut vec_znx_bigcoeff_t,
                a.cols() as u64,
            );
        }
    }
}
