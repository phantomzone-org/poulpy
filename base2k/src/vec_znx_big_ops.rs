use std::cmp::min;

use crate::ffi::vec_znx;
use crate::internals::{apply_binary_op, apply_unary_op, ffi_binary_op_factory_type_1, ffi_ternary_op_factory};
use crate::{Backend, FFT64, Module, VecZnx, VecZnxBig, VecZnxOps, ZnxBase, ZnxBasics, ZnxInfos, ZnxLayout, assert_alignement};

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
    fn vec_znx_big_normalize_tmp_bytes(&self, cols: usize) -> usize;

    /// Normalizes `a` and stores the result on `b`.
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: normalization basis.
    /// * `tmp_bytes`: scratch space of size at least [VecZnxBigOps::vec_znx_big_normalize].
    fn vec_znx_big_normalize(&self, log_base2k: usize, b: &mut VecZnx, a: &VecZnxBig<B>, tmp_bytes: &mut [u8]);

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

    fn vec_znx_big_normalize_tmp_bytes(&self, cols: usize) -> usize {
        Self::vec_znx_normalize_tmp_bytes(self, cols)
    }

    fn vec_znx_big_normalize(&self, log_base2k: usize, b: &mut VecZnx, a: &VecZnxBig<FFT64>, tmp_bytes: &mut [u8]) {
        #[cfg(debug_assertions)]
        {
            assert!(tmp_bytes.len() >= Self::vec_znx_big_normalize_tmp_bytes(&self, a.cols()));
            assert_alignement(tmp_bytes.as_ptr());
        }

        let a_size: usize = a.size();
        let b_size: usize = b.size();
        let a_sl: usize = a.sl();
        let b_sl: usize = b.sl();
        let a_cols: usize = a.cols();
        let b_cols: usize = b.cols();
        let min_cols: usize = min(a_cols, b_cols);
        (0..min_cols).for_each(|i| unsafe {
            vec_znx::vec_znx_normalize_base2k(
                self.ptr,
                log_base2k as u64,
                b.at_mut_ptr(i, 0),
                b_size as u64,
                b_sl as u64,
                a.at_ptr(i, 0),
                a_size as u64,
                a_sl as u64,
                tmp_bytes.as_mut_ptr(),
            );
        });

        (min_cols..b_cols).for_each(|i| (0..b_size).for_each(|j| b.zero_at(i, j)));
    }

    fn vec_znx_big_automorphism(&self, k: i64, b: &mut VecZnxBig<FFT64>, a: &VecZnxBig<FFT64>) {
        let op = ffi_binary_op_factory_type_1(
            self.ptr,
            k,
            b.size(),
            b.sl(),
            a.size(),
            a.sl(),
            vec_znx::vec_znx_automorphism,
        );
        apply_unary_op::<FFT64, VecZnxBig<FFT64>>(self, b, a, op);
    }

    fn vec_znx_big_automorphism_inplace(&self, k: i64, a: &mut VecZnxBig<FFT64>) {
        unsafe {
            let a_ptr: *mut VecZnxBig<FFT64> = a as *mut VecZnxBig<FFT64>;
            Self::vec_znx_big_automorphism(self, k, &mut *a_ptr, &*a_ptr);
        }
    }
}
