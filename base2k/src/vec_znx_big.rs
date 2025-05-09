use crate::ffi::vec_znx_big::{self, vec_znx_big_t};
use crate::{BACKEND, Infos, LAYOUT, Module, VecZnx, VecZnxDft, alloc_aligned, assert_alignement};

pub struct VecZnxBig {
    pub data: Vec<u8>,
    pub ptr: *mut u8,
    pub n: usize,
    pub size: usize,
    pub cols: usize,
    pub layout: LAYOUT,
    pub backend: BACKEND,
}

impl VecZnxBig {
    /// Returns a new [VecZnxBig] with the provided data as backing array.
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [Module::bytes_of_vec_znx_big].
    pub fn from_bytes(module: &Module, size: usize, cols: usize, bytes: &mut [u8]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert_eq!(bytes.len(), module.bytes_of_vec_znx_big(size, cols));
            assert_alignement(bytes.as_ptr())
        };
        unsafe {
            Self {
                data: Vec::from_raw_parts(bytes.as_mut_ptr(), bytes.len(), bytes.len()),
                ptr: bytes.as_mut_ptr(),
                n: module.n(),
                size: size,
                layout: LAYOUT::COL,
                cols: cols,
                backend: module.backend,
            }
        }
    }

    pub fn from_bytes_borrow(module: &Module, size: usize, cols: usize, bytes: &mut [u8]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert_eq!(bytes.len(), module.bytes_of_vec_znx_big(size, cols));
            assert_alignement(bytes.as_ptr());
        }
        Self {
            data: Vec::new(),
            ptr: bytes.as_mut_ptr(),
            n: module.n(),
            size: size,
            layout: LAYOUT::COL,
            cols: cols,
            backend: module.backend,
        }
    }

    pub fn as_vec_znx_dft(&mut self) -> VecZnxDft {
        VecZnxDft {
            data: Vec::new(),
            ptr: self.ptr,
            n: self.n,
            size: self.size,
            layout: LAYOUT::COL,
            cols: self.cols,
            backend: self.backend,
        }
    }

    pub fn backend(&self) -> BACKEND {
        self.backend
    }

    /// Returns a non-mutable reference of `T` of the entire contiguous array of the [VecZnxDft].
    /// When using [`crate::FFT64`] as backend, `T` should be [f64].
    /// When using [`crate::NTT120`] as backend, `T` should be [i64].
    /// The length of the returned array is cols * n.
    pub fn raw<T>(&self, module: &Module) -> &[T] {
        let ptr: *const T = self.ptr as *const T;
        let len: usize = (self.cols() * module.n() * 8) / std::mem::size_of::<T>();
        unsafe { &std::slice::from_raw_parts(ptr, len) }
    }
}

impl Infos for VecZnxBig {
    /// Returns the base 2 logarithm of the [VecZnx] degree.
    fn log_n(&self) -> usize {
        (usize::BITS - (self.n - 1).leading_zeros()) as _
    }

    /// Returns the [VecZnx] degree.
    fn n(&self) -> usize {
        self.n
    }

    fn size(&self) -> usize {
        self.size
    }

    fn layout(&self) -> LAYOUT {
        self.layout
    }

    /// Returns the number of cols of the [VecZnx].
    fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the number of rows of the [VecZnx].
    fn rows(&self) -> usize {
        1
    }
}

pub trait VecZnxBigOps {
    /// Allocates a vector Z[X]/(X^N+1) that stores not normalized values.
    fn new_vec_znx_big(&self, size: usize, cols: usize) -> VecZnxBig;

    /// Returns a new [VecZnxBig] with the provided bytes array as backing array.
    ///
    /// Behavior: takes ownership of the backing array.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of cols of the [VecZnxBig].
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_big].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_big].
    fn new_vec_znx_big_from_bytes(&self, size: usize, cols: usize, bytes: &mut [u8]) -> VecZnxBig;

    /// Returns a new [VecZnxBig] with the provided bytes array as backing array.
    ///
    /// Behavior: the backing array is only borrowed.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of cols of the [VecZnxBig].
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_big].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_big].
    fn new_vec_znx_big_from_bytes_borrow(&self, size: usize, cols: usize, tmp_bytes: &mut [u8]) -> VecZnxBig;

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnxBig] through [VecZnxBig::from_bytes].
    fn bytes_of_vec_znx_big(&self, size: usize, cols: usize) -> usize;

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
    fn vec_znx_big_normalize(&self, log_base2k: usize, b: &mut VecZnx, a: &VecZnxBig, tmp_bytes: &mut [u8]);

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
    fn new_vec_znx_big(&self, size: usize, cols: usize) -> VecZnxBig {
        let mut data: Vec<u8> = alloc_aligned::<u8>(self.bytes_of_vec_znx_big(size, cols));
        let ptr: *mut u8 = data.as_mut_ptr();
        VecZnxBig {
            data: data,
            ptr: ptr,
            n: self.n(),
            size: size,
            layout: LAYOUT::COL,
            cols: cols,
            backend: self.backend(),
        }
    }

    fn new_vec_znx_big_from_bytes(&self, size: usize, cols: usize, bytes: &mut [u8]) -> VecZnxBig {
        VecZnxBig::from_bytes(self, size, cols, bytes)
    }

    fn new_vec_znx_big_from_bytes_borrow(&self, size: usize, cols: usize, tmp_bytes: &mut [u8]) -> VecZnxBig {
        VecZnxBig::from_bytes_borrow(self, size, cols, tmp_bytes)
    }

    fn bytes_of_vec_znx_big(&self, size: usize, cols: usize) -> usize {
        unsafe { vec_znx_big::bytes_of_vec_znx_big(self.ptr, cols as u64) as usize * size }
    }

    fn vec_znx_big_sub_small_a_inplace(&self, b: &mut VecZnxBig, a: &VecZnx) {
        unsafe {
            vec_znx_big::vec_znx_big_sub_small_a(
                self.ptr,
                b.ptr as *mut vec_znx_big_t,
                b.cols() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
                b.ptr as *mut vec_znx_big_t,
                b.cols() as u64,
            )
        }
    }

    fn vec_znx_big_sub_small_a(&self, c: &mut VecZnxBig, a: &VecZnx, b: &VecZnxBig) {
        unsafe {
            vec_znx_big::vec_znx_big_sub_small_a(
                self.ptr,
                c.ptr as *mut vec_znx_big_t,
                c.cols() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
                b.ptr as *mut vec_znx_big_t,
                b.cols() as u64,
            )
        }
    }

    fn vec_znx_big_add_small(&self, c: &mut VecZnxBig, a: &VecZnx, b: &VecZnxBig) {
        unsafe {
            vec_znx_big::vec_znx_big_add_small(
                self.ptr,
                c.ptr as *mut vec_znx_big_t,
                c.cols() as u64,
                b.ptr as *mut vec_znx_big_t,
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
                b.ptr as *mut vec_znx_big_t,
                b.cols() as u64,
                b.ptr as *mut vec_znx_big_t,
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

    fn vec_znx_big_normalize(&self, log_base2k: usize, b: &mut VecZnx, a: &VecZnxBig, tmp_bytes: &mut [u8]) {
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
                a.ptr as *mut vec_znx_big_t,
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
                a.ptr as *mut vec_znx_big_t,
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
                b.ptr as *mut vec_znx_big_t,
                b.cols() as u64,
                a.ptr as *mut vec_znx_big_t,
                a.cols() as u64,
            );
        }
    }

    fn vec_znx_big_automorphism_inplace(&self, gal_el: i64, a: &mut VecZnxBig) {
        unsafe {
            vec_znx_big::vec_znx_big_automorphism(
                self.ptr,
                gal_el,
                a.ptr as *mut vec_znx_big_t,
                a.cols() as u64,
                a.ptr as *mut vec_znx_big_t,
                a.cols() as u64,
            );
        }
    }
}
