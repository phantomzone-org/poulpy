use crate::ffi::vec_znx_big::vec_znx_bigcoeff_t;
use crate::ffi::vec_znx_dft;
use crate::ffi::vec_znx_dft::{bytes_of_vec_znx_dft, vec_znx_dft_t};
use crate::{alloc_aligned, VecZnx};
use crate::{assert_alignement, Infos, Module, VecZnxBig, MODULETYPE};

pub struct VecZnxDft {
    pub data: Vec<u8>,
    pub ptr: *mut u8,
    pub n: usize,
    pub cols: usize,
    pub backend: MODULETYPE,
}

impl VecZnxDft {
    /// Returns a new [VecZnxDft] with the provided data as backing array.
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [Module::bytes_of_vec_znx_dft].
    pub fn from_bytes(module: &Module, cols: usize, bytes: &mut [u8]) -> VecZnxDft {
        #[cfg(debug_assertions)]
        {
            assert_eq!(bytes.len(), module.bytes_of_vec_znx_dft(cols));
            assert_alignement(bytes.as_ptr())
        }
        unsafe {
            VecZnxDft {
                data: Vec::from_raw_parts(bytes.as_mut_ptr(), bytes.len(), bytes.len()),
                ptr: bytes.as_mut_ptr(),
                n: module.n(),
                cols: cols,
                backend: module.backend,
            }
        }
    }

    pub fn from_bytes_borrow(module: &Module, cols: usize, bytes: &mut [u8]) -> VecZnxDft {
        #[cfg(debug_assertions)]
        {
            assert_eq!(bytes.len(), module.bytes_of_vec_znx_dft(cols));
            assert_alignement(bytes.as_ptr());
        }
        VecZnxDft {
            data: Vec::new(),
            ptr: bytes.as_mut_ptr(),
            n: module.n(),
            cols: cols,
            backend: module.backend,
        }
    }

    /// Cast a [VecZnxDft] into a [VecZnxBig].
    /// The returned [VecZnxBig] shares the backing array
    /// with the original [VecZnxDft].
    pub fn as_vec_znx_big(&mut self) -> VecZnxBig {
        VecZnxBig {
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

    /// Returns a non-mutable reference of `T` of the entire contiguous array of the [VecZnxDft].
    /// When using [`crate::FFT64`] as backend, `T` should be [f64].
    /// When using [`crate::NTT120`] as backend, `T` should be [i64].
    /// The length of the returned array is cols * n.
    pub fn raw<T>(&self, module: &Module) -> &[T] {
        let ptr: *const T = self.ptr as *const T;
        let len: usize = (self.cols() * module.n() * 8) / std::mem::size_of::<T>();
        unsafe { &std::slice::from_raw_parts(ptr, len) }
    }

    pub fn at<T>(&self, module: &Module, col_i: usize) -> &[T] {
        &self.raw::<T>(module)[col_i * module.n()..(col_i + 1) * module.n()]
    }

    /// Returns a mutable reference of `T` of the entire contiguous array of the [VecZnxDft].
    /// When using [`crate::FFT64`] as backend, `T` should be [f64].
    /// When using [`crate::NTT120`] as backend, `T` should be [i64].
    /// The length of the returned array is cols * n.
    pub fn raw_mut<T>(&self, module: &Module) -> &mut [T] {
        let ptr: *mut T = self.ptr as *mut T;
        let len: usize = (self.cols() * module.n() * 8) / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    pub fn at_mut<T>(&self, module: &Module, col_i: usize) -> &mut [T] {
        &mut self.raw_mut::<T>(module)[col_i * module.n()..(col_i + 1) * module.n()]
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

    fn vec_znx_dft(&self, b: &mut VecZnxDft, a: &VecZnx, a_limbs: usize);
}

impl VecZnxDftOps for Module {
    fn new_vec_znx_dft(&self, cols: usize) -> VecZnxDft {
        let mut data: Vec<u8> = alloc_aligned::<u8>(self.bytes_of_vec_znx_dft(cols));
        let ptr: *mut u8 = data.as_mut_ptr();
        VecZnxDft {
            data: data,
            ptr: ptr,
            n: self.n(),
            cols: cols,
            backend: self.backend(),
        }
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
        VecZnxDft::from_bytes(self, cols, tmp_bytes)
    }

    fn bytes_of_vec_znx_dft(&self, cols: usize) -> usize {
        unsafe { bytes_of_vec_znx_dft(self.ptr, cols as u64) as usize }
    }

    fn vec_znx_idft_tmp_a(&self, b: &mut VecZnxBig, a: &mut VecZnxDft, a_limbs: usize) {
        debug_assert!(
            b.cols() >= a_limbs,
            "invalid c_vector: b_vector.cols()={} < a_limbs={}",
            b.cols(),
            a_limbs
        );
        unsafe {
            vec_znx_dft::vec_znx_idft_tmp_a(
                self.ptr,
                b.ptr as *mut vec_znx_bigcoeff_t,
                b.cols() as u64,
                a.ptr as *mut vec_znx_dft_t,
                a_limbs as u64,
            )
        }
    }

    fn vec_znx_idft_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_dft::vec_znx_idft_tmp_bytes(self.ptr) as usize }
    }

    /// b <- DFT(a)
    ///
    /// # Panics
    /// If b.cols < a_cols
    fn vec_znx_dft(&self, b: &mut VecZnxDft, a: &VecZnx, a_cols: usize) {
        debug_assert!(
            b.cols() >= a_cols,
            "invalid a_cols: b.cols()={} < a_cols={}",
            b.cols(),
            a_cols
        );
        unsafe {
            vec_znx_dft::vec_znx_dft(
                self.ptr,
                b.ptr as *mut vec_znx_dft_t,
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
                self.ptr,
                b.ptr as *mut vec_znx_bigcoeff_t,
                a.cols() as u64,
                a.ptr as *mut vec_znx_dft_t,
                a_cols as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }
}
