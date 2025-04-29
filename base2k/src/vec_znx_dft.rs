use crate::ffi::vec_znx_big::vec_znx_big_t;
use crate::ffi::vec_znx_dft;
use crate::ffi::vec_znx_dft::{bytes_of_vec_znx_dft, vec_znx_dft_t};
use crate::{Backend, FFT64, Module, VecZnxBig, ZnxBase, ZnxInfos, ZnxLayout, assert_alignement};
use crate::{VecZnx, alloc_aligned};
use std::marker::PhantomData;

pub struct VecZnxDft<B: Backend> {
    pub data: Vec<u8>,
    pub ptr: *mut u8,
    pub n: usize,
    pub cols: usize,
    pub size: usize,
    pub _marker: PhantomData<B>,
}

impl<B: Backend> ZnxBase<B> for VecZnxDft<B> {
    type Scalar = u8;

    fn new(module: &Module<B>, cols: usize, size: usize) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(cols > 0);
            assert!(size > 0);
        }
        let mut data: Vec<Self::Scalar> = alloc_aligned(Self::bytes_of(module, cols, size));
        let ptr: *mut Self::Scalar = data.as_mut_ptr();
        Self {
            data: data,
            ptr: ptr,
            n: module.n(),
            size: size,
            cols: cols,
            _marker: PhantomData,
        }
    }

    fn bytes_of(module: &Module<B>, cols: usize, size: usize) -> usize {
        unsafe { bytes_of_vec_znx_dft(module.ptr, size as u64) as usize * cols }
    }

    /// Returns a new [VecZnxDft] with the provided data as backing array.
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [Module::bytes_of_vec_znx_dft].
    fn from_bytes(module: &Module<B>, cols: usize, size: usize, bytes: &mut [Self::Scalar]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(cols > 0);
            assert!(size > 0);
            assert_eq!(bytes.len(), Self::bytes_of(module, cols, size));
            assert_alignement(bytes.as_ptr())
        }
        unsafe {
            Self {
                data: Vec::from_raw_parts(bytes.as_mut_ptr(), bytes.len(), bytes.len()),
                ptr: bytes.as_mut_ptr(),
                n: module.n(),
                cols: cols,
                size: size,
                _marker: PhantomData,
            }
        }
    }

    fn from_bytes_borrow(module: &Module<B>, cols: usize, size: usize, bytes: &mut [Self::Scalar]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(cols > 0);
            assert!(size > 0);
            assert_eq!(bytes.len(), Self::bytes_of(module, cols, size));
            assert_alignement(bytes.as_ptr());
        }
        Self {
            data: Vec::new(),
            ptr: bytes.as_mut_ptr(),
            n: module.n(),
            cols: cols,
            size: size,
            _marker: PhantomData,
        }
    }
}

impl<B: Backend> VecZnxDft<B> {
    /// Cast a [VecZnxDft] into a [VecZnxBig].
    /// The returned [VecZnxBig] shares the backing array
    /// with the original [VecZnxDft].
    pub fn as_vec_znx_big(&mut self) -> VecZnxBig<B> {
        VecZnxBig::<B> {
            data: Vec::new(),
            ptr: self.ptr,
            n: self.n,
            cols: self.cols,
            size: self.size,
            _marker: PhantomData,
        }
    }
}

impl<B: Backend> ZnxInfos for VecZnxDft<B> {
    fn n(&self) -> usize {
        self.n
    }

    fn rows(&self) -> usize {
        1
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl ZnxLayout for VecZnxDft<FFT64> {
    type Scalar = f64;

    fn as_ptr(&self) -> *const Self::Scalar {
        self.ptr as *const Self::Scalar
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Scalar {
        self.ptr as *mut Self::Scalar
    }
}

impl VecZnxDft<FFT64> {
    pub fn print(&self, n: usize) {
        (0..self.size()).for_each(|i| println!("{}: {:?}", i, &self.at_limb(i)[..n]));
    }
}

pub trait VecZnxDftOps<B: Backend> {
    /// Allocates a vector Z[X]/(X^N+1) that stores normalized in the DFT space.
    fn new_vec_znx_dft(&self, cols: usize, size: usize) -> VecZnxDft<B>;

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
    fn new_vec_znx_dft_from_bytes(&self, cols: usize, size: usize, bytes: &mut [u8]) -> VecZnxDft<B>;

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
    fn new_vec_znx_dft_from_bytes_borrow(&self, cols: usize, size: usize, bytes: &mut [u8]) -> VecZnxDft<B>;

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

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnxDft] through [VecZnxDft::from_bytes].
    fn vec_znx_idft_tmp_bytes(&self) -> usize;

    /// b <- IDFT(a), uses a as scratch space.
    fn vec_znx_idft_tmp_a(&self, b: &mut VecZnxBig<B>, a: &mut VecZnxDft<B>);

    fn vec_znx_idft(&self, b: &mut VecZnxBig<B>, a: &VecZnxDft<B>, tmp_bytes: &mut [u8]);

    fn vec_znx_dft(&self, b: &mut VecZnxDft<B>, a: &VecZnx);

    fn vec_znx_dft_automorphism(&self, k: i64, b: &mut VecZnxDft<B>, a: &VecZnxDft<B>);

    fn vec_znx_dft_automorphism_inplace(&self, k: i64, a: &mut VecZnxDft<B>, tmp_bytes: &mut [u8]);

    fn vec_znx_dft_automorphism_tmp_bytes(&self) -> usize;
}

impl VecZnxDftOps<FFT64> for Module<FFT64> {
    fn new_vec_znx_dft(&self, cols: usize, size: usize) -> VecZnxDft<FFT64> {
        VecZnxDft::<FFT64>::new(&self, cols, size)
    }

    fn new_vec_znx_dft_from_bytes(&self, cols: usize, size: usize, tmp_bytes: &mut [u8]) -> VecZnxDft<FFT64> {
        VecZnxDft::from_bytes(self, cols, size, tmp_bytes)
    }

    fn new_vec_znx_dft_from_bytes_borrow(&self, cols: usize, size: usize, tmp_bytes: &mut [u8]) -> VecZnxDft<FFT64> {
        VecZnxDft::from_bytes_borrow(self, cols, size, tmp_bytes)
    }

    fn bytes_of_vec_znx_dft(&self, cols: usize, size: usize) -> usize {
        VecZnxDft::bytes_of(&self, cols, size)
    }

    fn vec_znx_idft_tmp_a(&self, b: &mut VecZnxBig<FFT64>, a: &mut VecZnxDft<FFT64>) {
        unsafe {
            vec_znx_dft::vec_znx_idft_tmp_a(
                self.ptr,
                b.ptr as *mut vec_znx_big_t,
                b.poly_count() as u64,
                a.ptr as *mut vec_znx_dft_t,
                a.poly_count() as u64,
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
    fn vec_znx_dft(&self, b: &mut VecZnxDft<FFT64>, a: &VecZnx) {
        unsafe {
            vec_znx_dft::vec_znx_dft(
                self.ptr,
                b.ptr as *mut vec_znx_dft_t,
                b.size() as u64,
                a.as_ptr(),
                a.size() as u64,
                (a.n() * a.cols()) as u64,
            )
        }
    }

    // b <- IDFT(a), scratch space size obtained with [vec_znx_idft_tmp_bytes].
    fn vec_znx_idft(&self, b: &mut VecZnxBig<FFT64>, a: &VecZnxDft<FFT64>, tmp_bytes: &mut [u8]) {
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
                b.poly_count() as u64,
                a.ptr as *const vec_znx_dft_t,
                a.poly_count() as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vec_znx_dft_automorphism(&self, k: i64, b: &mut VecZnxDft<FFT64>, a: &VecZnxDft<FFT64>) {
        unsafe {
            vec_znx_dft::vec_znx_dft_automorphism(
                self.ptr,
                k,
                b.ptr as *mut vec_znx_dft_t,
                b.poly_count() as u64,
                a.ptr as *const vec_znx_dft_t,
                a.poly_count() as u64,
                [0u8; 0].as_mut_ptr(),
            );
        }
    }

    fn vec_znx_dft_automorphism_inplace(&self, k: i64, a: &mut VecZnxDft<FFT64>, tmp_bytes: &mut [u8]) {
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
        println!("{}", a.poly_count());
        unsafe {
            vec_znx_dft::vec_znx_dft_automorphism(
                self.ptr,
                k,
                a.ptr as *mut vec_znx_dft_t,
                a.poly_count() as u64,
                a.ptr as *const vec_znx_dft_t,
                a.poly_count() as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vec_znx_dft_automorphism_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_dft::vec_znx_dft_automorphism_tmp_bytes(self.ptr) as usize }
    }
}

#[cfg(test)]
mod tests {
    use crate::{FFT64, Module, Sampling, VecZnx, VecZnxDft, VecZnxDftOps, VecZnxOps, ZnxLayout, alloc_aligned};
    use itertools::izip;
    use sampling::source::Source;

    #[test]
    fn test_automorphism_dft() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);

        let size: usize = 2;
        let log_base2k: usize = 17;
        let mut a: VecZnx = module.new_vec_znx(1, size);
        let mut a_dft: VecZnxDft<FFT64> = module.new_vec_znx_dft(1, size);
        let mut b_dft: VecZnxDft<FFT64> = module.new_vec_znx_dft(1, size);

        let mut source: Source = Source::new([0u8; 32]);
        module.fill_uniform(log_base2k, &mut a, 0, size, &mut source);

        let mut tmp_bytes: Vec<u8> = alloc_aligned(module.vec_znx_dft_automorphism_tmp_bytes());

        let p: i64 = -5;

        // a_dft <- DFT(a)
        module.vec_znx_dft(&mut a_dft, &a);

        // a_dft <- AUTO(a_dft)
        module.vec_znx_dft_automorphism_inplace(p, &mut a_dft, &mut tmp_bytes);

        // a <- AUTO(a)
        module.vec_znx_automorphism_inplace(p, &mut a, 0);

        // b_dft <- DFT(AUTO(a))
        module.vec_znx_dft(&mut b_dft, &a);

        let a_f64: &[f64] = a_dft.raw();
        let b_f64: &[f64] = b_dft.raw();
        izip!(a_f64.iter(), b_f64.iter()).for_each(|(ai, bi)| {
            assert!((ai - bi).abs() <= 1e-9, "{:+e} > 1e-9", (ai - bi).abs());
        });

        module.free()
    }
}
