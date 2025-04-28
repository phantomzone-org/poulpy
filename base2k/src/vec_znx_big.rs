use crate::ffi::vec_znx_big::{self, vec_znx_big_t};
use crate::{Backend, FFT64, Module, VecZnx, ZnxBase, ZnxInfos, ZnxLayout, alloc_aligned, assert_alignement};
use std::marker::PhantomData;

pub struct VecZnxBig<B: Backend> {
    pub data: Vec<u8>,
    pub ptr: *mut u8,
    pub n: usize,
    pub cols: usize,
    pub size: usize,
    pub _marker: PhantomData<B>,
}
impl<B: Backend> ZnxBase<B> for VecZnxBig<B> {
    type Scalar = u8;

    fn new(module: &Module<B>, cols: usize, size: usize) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(cols > 0);
            assert!(size > 0);
        }
        let mut data: Vec<Self::Scalar> = alloc_aligned::<u8>(Self::bytes_of(module, cols, size));
        let ptr: *mut Self::Scalar = data.as_mut_ptr();
        Self {
            data: data,
            ptr: ptr,
            n: module.n(),
            cols: cols,
            size: size,
            _marker: PhantomData,
        }
    }

    fn bytes_of(module: &Module<B>, cols: usize, size: usize) -> usize {
        unsafe { vec_znx_big::bytes_of_vec_znx_big(module.ptr, size as u64) as usize * cols }
    }

    /// Returns a new [VecZnxBig] with the provided data as backing array.
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [Module::bytes_of_vec_znx_big].
    fn from_bytes(module: &Module<B>, cols: usize, size: usize, bytes: &mut [Self::Scalar]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(cols > 0);
            assert!(size > 0);
            assert_eq!(bytes.len(), Self::bytes_of(module, cols, size));
            assert_alignement(bytes.as_ptr())
        };
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

impl<B: Backend> ZnxInfos for VecZnxBig<B> {
    fn n(&self) -> usize {
        self.n
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn rows(&self) -> usize {
        1
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl ZnxLayout for VecZnxBig<FFT64> {
    type Scalar = i64;

    fn as_ptr(&self) -> *const Self::Scalar {
        self.ptr as *const Self::Scalar
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Scalar {
        self.ptr as *mut Self::Scalar
    }
}

impl VecZnxBig<FFT64> {
    pub fn print(&self, n: usize) {
        (0..self.size()).for_each(|i| println!("{}: {:?}", i, &self.at_limb(i)[..n]));
    }
}

pub trait VecZnxBigOps<B: Backend> {
    /// Allocates a vector Z[X]/(X^N+1) that stores not normalized values.
    fn new_vec_znx_big(&self, cols: usize, size: usize) -> VecZnxBig<B>;

    /// Returns a new [VecZnxBig] with the provided bytes array as backing array.
    ///
    /// Behavior: takes ownership of the backing array.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of polynomials..
    /// * `size`: the number of size (a.k.a small polynomials) per polynomial.
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_big].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_big].
    fn new_vec_znx_big_from_bytes(&self, cols: usize, size: usize, bytes: &mut [u8]) -> VecZnxBig<B>;

    /// Returns a new [VecZnxBig] with the provided bytes array as backing array.
    ///
    /// Behavior: the backing array is only borrowed.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of polynomials..
    /// * `size`: the number of size (a.k.a small polynomials) per polynomial.
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_big].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_big].
    fn new_vec_znx_big_from_bytes_borrow(&self, cols: usize, size: usize, tmp_bytes: &mut [u8]) -> VecZnxBig<B>;

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnxBig] through [VecZnxBig::from_bytes].
    fn bytes_of_vec_znx_big(&self, cols: usize, size: usize) -> usize;

    /// b[VecZnxBig] <- b[VecZnxBig] - a[VecZnx]
    ///
    /// # Behavior
    ///
    /// [VecZnxBig] (3 cols and 4 size)
    /// [a0, b0, c0] [a1, b1, c1] [a2, b2, c2] [a3, b3, c3]
    /// -
    /// [VecZnx] (2 cols and 3 size)
    /// [d0, e0] [d1, e1] [d2, e2]
    /// =
    /// [a0-d0, b0-e0, c0] [a1-d1, b1-e1, c1] [a2-d2, b2-e2, c2] [a3, b3, c3]
    fn vec_znx_big_sub_small_a_inplace(&self, b: &mut VecZnxBig<B>, a: &VecZnx);

    /// c <- b - a
    fn vec_znx_big_sub_small_a(&self, c: &mut VecZnxBig<B>, a: &VecZnx, b: &VecZnxBig<B>);

    /// c <- b + a
    fn vec_znx_big_add_small(&self, c: &mut VecZnxBig<B>, a: &VecZnx, b: &VecZnxBig<B>);

    /// b <- b + a
    fn vec_znx_big_add_small_inplace(&self, b: &mut VecZnxBig<B>, a: &VecZnx);

    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize;

    /// b <- normalize(a)
    fn vec_znx_big_normalize(&self, log_base2k: usize, b: &mut VecZnx, a: &VecZnxBig<B>, tmp_bytes: &mut [u8]);

    fn vec_znx_big_range_normalize_base2k_tmp_bytes(&self) -> usize;

    fn vec_znx_big_range_normalize_base2k(
        &self,
        log_base2k: usize,
        res: &mut VecZnx,
        a: &VecZnxBig<B>,
        a_range_begin: usize,
        a_range_xend: usize,
        a_range_step: usize,
        tmp_bytes: &mut [u8],
    );

    fn vec_znx_big_automorphism(&self, gal_el: i64, b: &mut VecZnxBig<B>, a: &VecZnxBig<B>);

    fn vec_znx_big_automorphism_inplace(&self, gal_el: i64, a: &mut VecZnxBig<B>);
}

impl VecZnxBigOps<FFT64> for Module<FFT64> {
    fn new_vec_znx_big(&self, cols: usize, size: usize) -> VecZnxBig<FFT64> {
        VecZnxBig::new(self, cols, size)
    }

    fn new_vec_znx_big_from_bytes(&self, cols: usize, size: usize, bytes: &mut [u8]) -> VecZnxBig<FFT64> {
        VecZnxBig::from_bytes(self, cols, size, bytes)
    }

    fn new_vec_znx_big_from_bytes_borrow(&self, cols: usize, size: usize, tmp_bytes: &mut [u8]) -> VecZnxBig<FFT64> {
        VecZnxBig::from_bytes_borrow(self, cols, size, tmp_bytes)
    }

    fn bytes_of_vec_znx_big(&self, cols: usize, size: usize) -> usize {
        VecZnxBig::bytes_of(self, cols, size)
    }

    /// [VecZnxBig] (3 cols and 4 size)
    /// [a0, b0, c0] [a1, b1, c1] [a2, b2, c2] [a3, b3, c3]
    /// -
    /// [VecZnx] (2 cols and 3 size)
    /// [d0, e0] [d1, e1] [d2, e2]
    /// =
    /// [a0-d0, b0-e0, c0] [a1-d1, b1-e1, c1] [a2-d2, b2-e2, c2] [a3, b3, c3]
    fn vec_znx_big_sub_small_a_inplace(&self, b: &mut VecZnxBig<FFT64>, a: &VecZnx) {
        unsafe {
            vec_znx_big::vec_znx_big_sub_small_a(
                self.ptr,
                b.ptr as *mut vec_znx_big_t,
                b.poly_count() as u64,
                a.as_ptr(),
                a.poly_count() as u64,
                a.n() as u64,
                b.ptr as *mut vec_znx_big_t,
                b.poly_count() as u64,
            )
        }
    }

    fn vec_znx_big_sub_small_a(&self, c: &mut VecZnxBig<FFT64>, a: &VecZnx, b: &VecZnxBig<FFT64>) {
        unsafe {
            vec_znx_big::vec_znx_big_sub_small_a(
                self.ptr,
                c.ptr as *mut vec_znx_big_t,
                c.poly_count() as u64,
                a.as_ptr(),
                a.poly_count() as u64,
                a.n() as u64,
                b.ptr as *mut vec_znx_big_t,
                b.poly_count() as u64,
            )
        }
    }

    fn vec_znx_big_add_small(&self, c: &mut VecZnxBig<FFT64>, a: &VecZnx, b: &VecZnxBig<FFT64>) {
        unsafe {
            vec_znx_big::vec_znx_big_add_small(
                self.ptr,
                c.ptr as *mut vec_znx_big_t,
                c.poly_count() as u64,
                b.ptr as *mut vec_znx_big_t,
                b.poly_count() as u64,
                a.as_ptr(),
                a.poly_count() as u64,
                a.n() as u64,
            )
        }
    }

    fn vec_znx_big_add_small_inplace(&self, b: &mut VecZnxBig<FFT64>, a: &VecZnx) {
        unsafe {
            vec_znx_big::vec_znx_big_add_small(
                self.ptr,
                b.ptr as *mut vec_znx_big_t,
                b.poly_count() as u64,
                b.ptr as *mut vec_znx_big_t,
                b.poly_count() as u64,
                a.as_ptr(),
                a.poly_count() as u64,
                a.n() as u64,
            )
        }
    }

    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize {
        unsafe { vec_znx_big::vec_znx_big_normalize_base2k_tmp_bytes(self.ptr) as usize }
    }

    fn vec_znx_big_normalize(&self, log_base2k: usize, b: &mut VecZnx, a: &VecZnxBig<FFT64>, tmp_bytes: &mut [u8]) {
        debug_assert!(
            tmp_bytes.len() >= Self::vec_znx_big_normalize_tmp_bytes(self),
            "invalid tmp_bytes: tmp_bytes.len()={} <= self.vec_znx_big_normalize_tmp_bytes()={}",
            tmp_bytes.len(),
            Self::vec_znx_big_normalize_tmp_bytes(self)
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
                b.size() as u64,
                b.n() as u64,
                a.ptr as *mut vec_znx_big_t,
                a.size() as u64,
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
        a: &VecZnxBig<FFT64>,
        a_range_begin: usize,
        a_range_xend: usize,
        a_range_step: usize,
        tmp_bytes: &mut [u8],
    ) {
        debug_assert!(
            tmp_bytes.len() >= Self::vec_znx_big_range_normalize_base2k_tmp_bytes(self),
            "invalid tmp_bytes: tmp_bytes.len()={} <= self.vec_znx_big_range_normalize_base2k_tmp_bytes()={}",
            tmp_bytes.len(),
            Self::vec_znx_big_range_normalize_base2k_tmp_bytes(self)
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
                res.size() as u64,
                res.n() as u64,
                a.ptr as *mut vec_znx_big_t,
                a_range_begin as u64,
                a_range_xend as u64,
                a_range_step as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vec_znx_big_automorphism(&self, gal_el: i64, b: &mut VecZnxBig<FFT64>, a: &VecZnxBig<FFT64>) {
        unsafe {
            vec_znx_big::vec_znx_big_automorphism(
                self.ptr,
                gal_el,
                b.ptr as *mut vec_znx_big_t,
                b.poly_count() as u64,
                a.ptr as *mut vec_znx_big_t,
                a.poly_count() as u64,
            );
        }
    }

    fn vec_znx_big_automorphism_inplace(&self, gal_el: i64, a: &mut VecZnxBig<FFT64>) {
        unsafe {
            vec_znx_big::vec_znx_big_automorphism(
                self.ptr,
                gal_el,
                a.ptr as *mut vec_znx_big_t,
                a.poly_count() as u64,
                a.ptr as *mut vec_znx_big_t,
                a.poly_count() as u64,
            );
        }
    }
}
