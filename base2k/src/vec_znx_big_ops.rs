use crate::ffi::vec_znx_big::vec_znx_big_t;
use crate::ffi::{vec_znx, vec_znx_big};
use crate::internals::{apply_binary_op, ffi_ternary_op_factory};
use crate::{Backend, FFT64, Module, VecZnx, VecZnxBig, ZnxBase, ZnxInfos, ZnxLayout, assert_alignement};

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
    /// * `size`: the number of polynomials per column.
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
    /// * `size`: the number of polynomials per column.
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_big].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_big].
    fn new_vec_znx_big_from_bytes_borrow(&self, cols: usize, size: usize, tmp_bytes: &mut [u8]) -> VecZnxBig<B>;

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnxBig] through [VecZnxBig::from_bytes].
    fn bytes_of_vec_znx_big(&self, cols: usize, size: usize) -> usize;

    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add(&self, c: &mut VecZnxBig<B>, a: &VecZnxBig<B>, b: &VecZnxBig<B>);

    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_inplace(&self, b: &mut VecZnxBig<B>, a: &VecZnxBig<B>);

    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_small(&self, c: &mut VecZnxBig<B>, a: &VecZnx, b: &VecZnxBig<B>);

    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_small_inplace(&self, b: &mut VecZnxBig<B>, a: &VecZnx);

    /// Subtracts `a` to `b` and stores the result on `c`.
    fn vec_znx_big_sub(&self, c: &mut VecZnxBig<B>, a: &VecZnxBig<B>, b: &VecZnxBig<B>);

    /// Subtracts `a` to `b` and stores the result on `b`.
    fn vec_znx_big_sub_ab_inplace(&self, b: &mut VecZnxBig<B>, a: &VecZnxBig<B>);

    /// Subtracts `b` to `a` and stores the result on `b`.
    fn vec_znx_big_sub_ba_inplace(&self, b: &mut VecZnxBig<B>, a: &VecZnxBig<B>);

    /// Subtracts `b` to `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_ab(&self, c: &mut VecZnxBig<B>, a: &VecZnx, b: &VecZnxBig<B>);

    /// Subtracts `a` to `b` and stores the result on `b`.
    fn vec_znx_big_sub_small_ab_inplace(&self, b: &mut VecZnxBig<B>, a: &VecZnx);

    /// Subtracts `b` to `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_ba(&self, c: &mut VecZnxBig<B>, a: &VecZnxBig<B>, b: &VecZnx);

    /// Subtracts `b` to `a` and stores the result on `b`.
    fn vec_znx_big_sub_small_ba_inplace(&self, b: &mut VecZnxBig<B>, a: &VecZnx);

    /// Returns the minimum number of bytes to apply [VecZnxBigOps::vec_znx_big_normalize].
    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize;

    /// Normalizes `a` and stores the result on `b`.
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: normalization basis.
    /// * `tmp_bytes`: scratch space of size at least [VecZnxBigOps::vec_znx_big_normalize].
    fn vec_znx_big_normalize(&self, log_base2k: usize, b: &mut VecZnx, a: &VecZnxBig<B>, tmp_bytes: &mut [u8]);

    /// Returns the minimum number of bytes to apply [VecZnxBigOps::vec_znx_big_range_normalize_base2k].
    fn vec_znx_big_range_normalize_base2k_tmp_bytes(&self) -> usize;

    /// Normalize `a`, taking into account column interleaving and stores the result on `b`.
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: normalization basis.
    /// * `a_range_begin`: column to start.
    /// * `a_range_end`: column to end.
    /// * `a_range_step`: column step size.
    /// * `tmp_bytes`: scratch space of size at least [VecZnxBigOps::vec_znx_big_range_normalize_base2k_tmp_bytes].
    fn vec_znx_big_range_normalize_base2k(
        &self,
        log_base2k: usize,
        b: &mut VecZnx,
        a: &VecZnxBig<B>,
        a_range_begin: usize,
        a_range_xend: usize,
        a_range_step: usize,
        tmp_bytes: &mut [u8],
    );

    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `b`.
    fn vec_znx_big_automorphism(&self, k: i64, b: &mut VecZnxBig<B>, a: &VecZnxBig<B>);

    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `a`.
    fn vec_znx_big_automorphism_inplace(&self, k: i64, a: &mut VecZnxBig<B>);
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

    fn vec_znx_big_add(&self, c: &mut VecZnxBig<FFT64>, a: &VecZnxBig<FFT64>, b: &VecZnxBig<FFT64>) {
        let op = ffi_ternary_op_factory(
            self.ptr,
            c.size(),
            c.sl(),
            a.size(),
            a.sl(),
            b.size(),
            b.sl(),
            vec_znx::vec_znx_add,
        );
        apply_binary_op::<FFT64, VecZnxBig<FFT64>, VecZnxBig<FFT64>, VecZnxBig<FFT64>, false>(self, c, a, b, op);
    }

    fn vec_znx_big_add_inplace(&self, b: &mut VecZnxBig<FFT64>, a: &VecZnxBig<FFT64>) {
        unsafe {
            let b_ptr: *mut VecZnxBig<FFT64> = b as *mut VecZnxBig<FFT64>;
            Self::vec_znx_big_add(self, &mut *b_ptr, a, &*b_ptr);
        }
    }

    fn vec_znx_big_sub(&self, c: &mut VecZnxBig<FFT64>, a: &VecZnxBig<FFT64>, b: &VecZnxBig<FFT64>) {
        let op = ffi_ternary_op_factory(
            self.ptr,
            c.size(),
            c.sl(),
            a.size(),
            a.sl(),
            b.size(),
            b.sl(),
            vec_znx::vec_znx_sub,
        );
        apply_binary_op::<FFT64, VecZnxBig<FFT64>, VecZnxBig<FFT64>, VecZnxBig<FFT64>, true>(self, c, a, b, op);
    }

    fn vec_znx_big_sub_ab_inplace(&self, b: &mut VecZnxBig<FFT64>, a: &VecZnxBig<FFT64>) {
        unsafe {
            let b_ptr: *mut VecZnxBig<FFT64> = b as *mut VecZnxBig<FFT64>;
            Self::vec_znx_big_sub(self, &mut *b_ptr, a, &*b_ptr);
        }
    }

    fn vec_znx_big_sub_ba_inplace(&self, b: &mut VecZnxBig<FFT64>, a: &VecZnxBig<FFT64>) {
        unsafe {
            let b_ptr: *mut VecZnxBig<FFT64> = b as *mut VecZnxBig<FFT64>;
            Self::vec_znx_big_sub(self, &mut *b_ptr, &*b_ptr, a);
        }
    }

    fn vec_znx_big_sub_small_ba(&self, c: &mut VecZnxBig<FFT64>, a: &VecZnxBig<FFT64>, b: &VecZnx) {
        let op = ffi_ternary_op_factory(
            self.ptr,
            c.size(),
            c.sl(),
            a.size(),
            a.sl(),
            b.size(),
            b.sl(),
            vec_znx::vec_znx_sub,
        );
        apply_binary_op::<FFT64, VecZnxBig<FFT64>, VecZnxBig<FFT64>, VecZnx, true>(self, c, a, b, op);
    }

    fn vec_znx_big_sub_small_ba_inplace(&self, b: &mut VecZnxBig<FFT64>, a: &VecZnx) {
        unsafe {
            let b_ptr: *mut VecZnxBig<FFT64> = b as *mut VecZnxBig<FFT64>;
            Self::vec_znx_big_sub_small_ba(self, &mut *b_ptr, &*b_ptr, a);
        }
    }

    fn vec_znx_big_sub_small_ab(&self, c: &mut VecZnxBig<FFT64>, a: &VecZnx, b: &VecZnxBig<FFT64>) {
        let op = ffi_ternary_op_factory(
            self.ptr,
            c.size(),
            c.sl(),
            a.size(),
            a.sl(),
            b.size(),
            b.sl(),
            vec_znx::vec_znx_sub,
        );
        apply_binary_op::<FFT64, VecZnxBig<FFT64>, VecZnx, VecZnxBig<FFT64>, true>(self, c, a, b, op);
    }

    fn vec_znx_big_sub_small_ab_inplace(&self, b: &mut VecZnxBig<FFT64>, a: &VecZnx) {
        unsafe {
            let b_ptr: *mut VecZnxBig<FFT64> = b as *mut VecZnxBig<FFT64>;
            Self::vec_znx_big_sub_small_ab(self, &mut *b_ptr, a, &*b_ptr);
        }
    }

    fn vec_znx_big_add_small(&self, c: &mut VecZnxBig<FFT64>, a: &VecZnx, b: &VecZnxBig<FFT64>) {
        let op = ffi_ternary_op_factory(
            self.ptr,
            c.size(),
            c.sl(),
            a.size(),
            a.sl(),
            b.size(),
            b.sl(),
            vec_znx::vec_znx_add,
        );
        apply_binary_op::<FFT64, VecZnxBig<FFT64>, VecZnx, VecZnxBig<FFT64>, false>(self, c, a, b, op);
    }

    fn vec_znx_big_add_small_inplace(&self, b: &mut VecZnxBig<FFT64>, a: &VecZnx) {
        unsafe {
            let b_ptr: *mut VecZnxBig<FFT64> = b as *mut VecZnxBig<FFT64>;
            Self::vec_znx_big_add_small(self, &mut *b_ptr, a, &*b_ptr);
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
