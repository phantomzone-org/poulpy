use crate::ffi::vec_znx_big::vec_znx_big_t;
use crate::ffi::vec_znx_dft;
use crate::ffi::vec_znx_dft::{bytes_of_vec_znx_dft, vec_znx_dft_t};
use crate::{BACKEND, Infos, LAYOUT, Module, VecZnxBig, assert_alignement};
use crate::{DEFAULTALIGN, VecZnx, alloc_aligned};

pub struct VecZnxDft {
    pub data: Vec<u8>,
    pub ptr: *mut u8,
    pub n: usize,
    pub size: usize,
    pub layout: LAYOUT,
    pub cols: usize,
    pub backend: BACKEND,
}

impl VecZnxDft {
    /// Returns a new [VecZnxDft] with the provided data as backing array.
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [Module::bytes_of_vec_znx_dft].
    pub fn from_bytes(module: &Module, size: usize, cols: usize, bytes: &mut [u8]) -> VecZnxDft {
        #[cfg(debug_assertions)]
        {
            assert_eq!(bytes.len(), module.bytes_of_vec_znx_dft(size, cols));
            assert_alignement(bytes.as_ptr())
        }
        unsafe {
            VecZnxDft {
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

    pub fn from_bytes_borrow(module: &Module, size: usize, cols: usize, bytes: &mut [u8]) -> VecZnxDft {
        #[cfg(debug_assertions)]
        {
            assert_eq!(bytes.len(), module.bytes_of_vec_znx_dft(size, cols));
            assert_alignement(bytes.as_ptr());
        }
        VecZnxDft {
            data: Vec::new(),
            ptr: bytes.as_mut_ptr(),
            n: module.n(),
            size: size,
            layout: LAYOUT::COL,
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
            layout: LAYOUT::COL,
            size: self.size,
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

impl Infos for VecZnxDft {
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

pub trait VecZnxDftOps {
    /// Allocates a vector Z[X]/(X^N+1) that stores normalized in the DFT space.
    fn new_vec_znx_dft(&self, size: usize, cols: usize) -> VecZnxDft;

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
    fn new_vec_znx_dft_from_bytes(&self, size: usize, cols: usize, bytes: &mut [u8]) -> VecZnxDft;

    /// Returns a new [VecZnxDft] with the provided bytes array as backing array.
    ///
    /// Behavior: the backing array is only borrowed.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of cols of the [VecZnxDft].
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_dft].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_dft].
    fn new_vec_znx_dft_from_bytes_borrow(&self, size: usize, cols: usize, bytes: &mut [u8]) -> VecZnxDft;

    /// Returns a new [VecZnxDft] with the provided bytes array as backing array.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of cols of the [VecZnxDft].
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_dft].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_dft].
    fn bytes_of_vec_znx_dft(&self, size: usize, cols: usize) -> usize;

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnxDft] through [VecZnxDft::from_bytes].
    fn vec_znx_idft_tmp_bytes(&self) -> usize;

    /// b <- IDFT(a), uses a as scratch space.
    fn vec_znx_idft_tmp_a(&self, b: &mut VecZnxBig, a: &mut VecZnxDft);

    fn vec_znx_idft(&self, b: &mut VecZnxBig, a: &VecZnxDft, tmp_bytes: &mut [u8]);

    fn vec_znx_dft(&self, b: &mut VecZnxDft, a: &VecZnx);

    fn vec_znx_dft_automorphism(&self, k: i64, b: &mut VecZnxDft, a: &VecZnxDft);

    fn vec_znx_dft_automorphism_inplace(&self, k: i64, a: &mut VecZnxDft, tmp_bytes: &mut [u8]);

    fn vec_znx_dft_automorphism_tmp_bytes(&self) -> usize;
}

impl VecZnxDftOps for Module {
    fn new_vec_znx_dft(&self, size: usize, cols: usize) -> VecZnxDft {
        let mut data: Vec<u8> = alloc_aligned::<u8>(self.bytes_of_vec_znx_dft(size, cols));
        let ptr: *mut u8 = data.as_mut_ptr();
        VecZnxDft {
            data: data,
            ptr: ptr,
            n: self.n(),
            size: size,
            layout: LAYOUT::COL,
            cols: cols,
            backend: self.backend(),
        }
    }

    fn new_vec_znx_dft_from_bytes(&self, size: usize, cols: usize, tmp_bytes: &mut [u8]) -> VecZnxDft {
        VecZnxDft::from_bytes(self, size, cols, tmp_bytes)
    }

    fn new_vec_znx_dft_from_bytes_borrow(&self, size: usize, cols: usize, tmp_bytes: &mut [u8]) -> VecZnxDft {
        VecZnxDft::from_bytes_borrow(self, size, cols, tmp_bytes)
    }

    fn bytes_of_vec_znx_dft(&self, size: usize, cols: usize) -> usize {
        unsafe { bytes_of_vec_znx_dft(self.ptr, cols as u64) as usize * size }
    }

    fn vec_znx_idft_tmp_a(&self, b: &mut VecZnxBig, a: &mut VecZnxDft) {
        unsafe {
            vec_znx_dft::vec_znx_idft_tmp_a(
                self.ptr,
                b.ptr as *mut vec_znx_big_t,
                b.cols() as u64,
                a.ptr as *mut vec_znx_dft_t,
                a.cols() as u64,
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
    fn vec_znx_dft(&self, b: &mut VecZnxDft, a: &VecZnx) {
        unsafe {
            vec_znx_dft::vec_znx_dft(
                self.ptr,
                b.ptr as *mut vec_znx_dft_t,
                b.cols() as u64,
                a.as_ptr(),
                a.cols() as u64,
                a.n() as u64,
            )
        }
    }

    // b <- IDFT(a), scratch space size obtained with [vec_znx_idft_tmp_bytes].
    fn vec_znx_idft(&self, b: &mut VecZnxBig, a: &VecZnxDft, tmp_bytes: &mut [u8]) {
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
        unsafe {
            vec_znx_dft::vec_znx_idft(
                self.ptr,
                b.ptr as *mut vec_znx_big_t,
                b.cols() as u64,
                a.ptr as *const vec_znx_dft_t,
                a.cols() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vec_znx_dft_automorphism(&self, k: i64, b: &mut VecZnxDft, a: &VecZnxDft) {
        unsafe {
            vec_znx_dft::vec_znx_dft_automorphism(
                self.ptr,
                k,
                b.ptr as *mut vec_znx_dft_t,
                b.cols() as u64,
                a.ptr as *const vec_znx_dft_t,
                a.cols() as u64,
                [0u8; 0].as_mut_ptr(),
            );
        }
    }

    fn vec_znx_dft_automorphism_inplace(&self, k: i64, a: &mut VecZnxDft, tmp_bytes: &mut [u8]) {
        #[cfg(debug_assertions)]
        {
            assert!(
                tmp_bytes.len() >= Self::vec_znx_dft_automorphism_tmp_bytes(self),
                "invalid tmp_bytes: tmp_bytes.len()={} < self.vec_znx_dft_automorphism_tmp_bytes()={}",
                tmp_bytes.len(),
                Self::vec_znx_dft_automorphism_tmp_bytes(self)
            );
            assert_alignement(tmp_bytes.as_ptr())
        }
        unsafe {
            vec_znx_dft::vec_znx_dft_automorphism(
                self.ptr,
                k,
                a.ptr as *mut vec_znx_dft_t,
                a.cols() as u64,
                a.ptr as *const vec_znx_dft_t,
                a.cols() as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vec_znx_dft_automorphism_tmp_bytes(&self) -> usize {
        unsafe {
            std::cmp::max(
                vec_znx_dft::vec_znx_dft_automorphism_tmp_bytes(self.ptr) as usize,
                DEFAULTALIGN,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{BACKEND, Module, Sampling, VecZnx, VecZnxDft, VecZnxDftOps, VecZnxOps, alloc_aligned};
    use itertools::izip;
    use sampling::source::{Source, new_seed};

    #[test]
    fn test_automorphism_dft() {
        let module: Module = Module::new(128, BACKEND::FFT64);

        let cols: usize = 2;
        let log_base2k: usize = 17;
        let mut a: VecZnx = module.new_vec_znx(1, cols);
        let mut a_dft: VecZnxDft = module.new_vec_znx_dft(1, cols);
        let mut b_dft: VecZnxDft = module.new_vec_znx_dft(1, cols);

        let mut source: Source = Source::new(new_seed());
        module.fill_uniform(log_base2k, &mut a, cols, &mut source);

        let mut tmp_bytes: Vec<u8> = alloc_aligned(module.vec_znx_dft_automorphism_tmp_bytes());

        let p: i64 = -5;

        // a_dft <- DFT(a)
        module.vec_znx_dft(&mut a_dft, &a);

        // a_dft <- AUTO(a_dft)
        module.vec_znx_dft_automorphism_inplace(p, &mut a_dft, &mut tmp_bytes);

        // a <- AUTO(a)
        module.vec_znx_automorphism_inplace(p, &mut a);

        // b_dft <- DFT(AUTO(a))
        module.vec_znx_dft(&mut b_dft, &a);

        let a_f64: &[f64] = a_dft.raw(&module);
        let b_f64: &[f64] = b_dft.raw(&module);
        izip!(a_f64.iter(), b_f64.iter()).for_each(|(ai, bi)| {
            assert!((ai - bi).abs() <= 1e-9, "{:+e} > 1e-9", (ai - bi).abs());
        });

        module.free()
    }
}
