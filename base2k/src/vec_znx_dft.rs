use crate::ffi::vec_znx_big::vec_znx_big_t;
use crate::ffi::vec_znx_dft;
use crate::ffi::vec_znx_dft::{bytes_of_vec_znx_dft, vec_znx_dft_t};
use crate::{Backend, FFT64, Infos, Module, VecZnxBig, assert_alignement};
use crate::{DEFAULTALIGN, VecZnx, alloc_aligned};
use std::marker::PhantomData;

pub struct VecZnxDft<B: Backend> {
    pub data: Vec<u8>,
    pub ptr: *mut u8,
    pub n: usize,
    pub cols: usize,
    pub limbs: usize,
    pub _marker: PhantomData<B>,
}

impl VecZnxDft<FFT64> {
    pub fn new(module: &Module<FFT64>, cols: usize, limbs: usize) -> Self {
        let mut data: Vec<u8> = alloc_aligned::<u8>(module.bytes_of_vec_znx_dft(cols, limbs));
        let ptr: *mut u8 = data.as_mut_ptr();
        Self {
            data: data,
            ptr: ptr,
            n: module.n(),
            limbs: limbs,
            cols: cols,
            _marker: PhantomData,
        }
    }
    /// Returns a new [VecZnxDft] with the provided data as backing array.
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [Module::bytes_of_vec_znx_dft].
    pub fn from_bytes(module: &Module<FFT64>, cols: usize, limbs: usize, bytes: &mut [u8]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(cols > 0);
            assert!(limbs > 0);
            assert_eq!(bytes.len(), module.bytes_of_vec_znx_dft(cols, limbs));
            assert_alignement(bytes.as_ptr())
        }
        unsafe {
            Self {
                data: Vec::from_raw_parts(bytes.as_mut_ptr(), bytes.len(), bytes.len()),
                ptr: bytes.as_mut_ptr(),
                n: module.n(),
                cols: cols,
                limbs: limbs,
                _marker: PhantomData,
            }
        }
    }

    pub fn from_bytes_borrow(module: &Module<FFT64>, cols: usize, limbs: usize, bytes: &mut [u8]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(cols > 0);
            assert!(limbs > 0);
            assert_eq!(bytes.len(), module.bytes_of_vec_znx_dft(cols, limbs));
            assert_alignement(bytes.as_ptr());
        }
        Self {
            data: Vec::new(),
            ptr: bytes.as_mut_ptr(),
            n: module.n(),
            cols: cols,
            limbs: limbs,
            _marker: PhantomData,
        }
    }

    /// Cast a [VecZnxDft] into a [VecZnxBig].
    /// The returned [VecZnxBig] shares the backing array
    /// with the original [VecZnxDft].
    pub fn as_vec_znx_big(&mut self) -> VecZnxBig<FFT64> {
        VecZnxBig::<FFT64> {
            data: Vec::new(),
            ptr: self.ptr,
            n: self.n,
            cols: self.cols,
            limbs: self.limbs,
            _marker: PhantomData,
        }
    }

    /// Returns a non-mutable pointer to the backedn slice of the receiver.
    pub fn as_ptr(&self) -> *const f64 {
        self.ptr as *const f64
    }

    /// Returns a mutable pointer to the backedn slice of the receiver.
    pub fn as_mut_ptr(&mut self) -> *mut f64 {
        self.ptr as *mut f64
    }

    pub fn raw(&self) -> &[f64] {
        unsafe { &std::slice::from_raw_parts(self.as_ptr(), self.n() * self.poly_count()) }
    }

    pub fn raw_mut(&mut self) -> &mut [f64] {
        let ptr: *mut f64 = self.ptr as *mut f64;
        let size: usize = self.n() * self.poly_count();
        unsafe { std::slice::from_raw_parts_mut(ptr, size) }
    }

    pub fn at_ptr(&self, i: usize, j: usize) -> *const f64 {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.cols());
            assert!(j < self.limbs());
        }
        let offset: usize = self.n * (j * self.cols() + i);
        self.as_ptr().wrapping_add(offset)
    }

    /// Returns a non-mutable reference to the i-th limb.
    /// The returned array is of size [Self::n()] * [Self::cols()].
    pub fn at_limb(&self, i: usize) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.at_ptr(0, i), self.n * self.cols()) }
    }

    /// Returns a non-mutable reference to the (i, j)-th poly.
    /// The returned array is of size [Self::n()].
    pub fn at_poly(&self, i: usize, j: usize) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.at_ptr(i, j), self.n) }
    }

    /// Returns a mutable pointer starting a the (i, j)-th small poly.
    pub fn at_mut_ptr(&mut self, i: usize, j: usize) -> *mut f64 {
        #[cfg(debug_assertions)]
        {
            assert!(i < self.cols());
            assert!(j < self.limbs());
        }

        let offset: usize = self.n * (j * self.cols() + i);
        self.as_mut_ptr().wrapping_add(offset)
    }

    /// Returns a mutable reference to the i-th limb.
    /// The returned array is of size [Self::n()] * [Self::cols()].
    pub fn at_limb_mut(&mut self, i: usize) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.at_mut_ptr(0, i), self.n * self.cols()) }
    }

    /// Returns a mutable reference to the (i, j)-th poly.
    /// The returned array is of size [Self::n()].
    pub fn at_poly_mut(&mut self, i: usize, j: usize) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.at_mut_ptr(i, j), self.n) }
    }

    pub fn print(&self, n: usize) {
        (0..self.limbs()).for_each(|i| println!("{}: {:?}", i, &self.at_limb(i)[..n]));
    }
}

impl<B: Backend> Infos for VecZnxDft<B> {
    fn n(&self) -> usize {
        self.n
    }

    fn log_n(&self) -> usize {
        (usize::BITS - (self.n() - 1).leading_zeros()) as _
    }

    fn rows(&self) -> usize {
        1
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn limbs(&self) -> usize {
        self.limbs
    }

    fn poly_count(&self) -> usize {
        self.cols * self.limbs
    }
}

pub trait VecZnxDftOps<B: Backend> {
    /// Allocates a vector Z[X]/(X^N+1) that stores normalized in the DFT space.
    fn new_vec_znx_dft(&self, cols: usize, limbs: usize) -> VecZnxDft<B>;

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
    fn new_vec_znx_dft_from_bytes(&self, cols: usize, limbs: usize, bytes: &mut [u8]) -> VecZnxDft<B>;

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
    fn new_vec_znx_dft_from_bytes_borrow(&self, cols: usize, limbs: usize, bytes: &mut [u8]) -> VecZnxDft<B>;

    /// Returns a new [VecZnxDft] with the provided bytes array as backing array.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of cols of the [VecZnxDft].
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_dft].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_dft].
    fn bytes_of_vec_znx_dft(&self, cols: usize, limbs: usize) -> usize;

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
    fn new_vec_znx_dft(&self, cols: usize, limbs: usize) -> VecZnxDft<FFT64> {
        VecZnxDft::<FFT64>::new(&self, cols, limbs)
    }

    fn new_vec_znx_dft_from_bytes(&self, cols: usize, limbs: usize, tmp_bytes: &mut [u8]) -> VecZnxDft<FFT64> {
        VecZnxDft::from_bytes(self, cols, limbs, tmp_bytes)
    }

    fn new_vec_znx_dft_from_bytes_borrow(&self, cols: usize, limbs: usize, tmp_bytes: &mut [u8]) -> VecZnxDft<FFT64> {
        VecZnxDft::from_bytes_borrow(self, cols, limbs, tmp_bytes)
    }

    fn bytes_of_vec_znx_dft(&self, cols: usize, limbs: usize) -> usize {
        unsafe { bytes_of_vec_znx_dft(self.ptr, limbs as u64) as usize * cols }
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
                b.limbs() as u64,
                a.as_ptr(),
                a.limbs() as u64,
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
    use crate::{FFT64, Module, Sampling, VecZnx, VecZnxDft, VecZnxDftOps, VecZnxOps, alloc_aligned};
    use itertools::izip;
    use sampling::source::Source;

    #[test]
    fn test_automorphism_dft() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);

        let limbs: usize = 2;
        let log_base2k: usize = 17;
        let mut a: VecZnx = module.new_vec_znx(1, limbs);
        let mut a_dft: VecZnxDft<FFT64> = module.new_vec_znx_dft(1, limbs);
        let mut b_dft: VecZnxDft<FFT64> = module.new_vec_znx_dft(1, limbs);

        let mut source: Source = Source::new([0u8; 32]);
        module.fill_uniform(log_base2k, &mut a, 0, limbs, &mut source);

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

        let a_f64: &[f64] = a_dft.raw();
        let b_f64: &[f64] = b_dft.raw();
        izip!(a_f64.iter(), b_f64.iter()).for_each(|(ai, bi)| {
            assert!((ai - bi).abs() <= 1e-9, "{:+e} > 1e-9", (ai - bi).abs());
        });

        module.free()
    }
}
