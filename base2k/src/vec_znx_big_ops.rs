use crate::ffi::vec_znx;
use crate::{Backend, FFT64, Module, VecZnx, VecZnxBig, VecZnxOps, ZnxBase, ZnxInfos, ZnxLayout, assert_alignement};

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
    fn vec_znx_big_add(
        &self,
        res: &mut VecZnxBig<B>,
        col_res: usize,
        a: &VecZnxBig<B>,
        col_a: usize,
        b: &VecZnxBig<B>,
        col_b: usize,
    );

    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_inplace(&self, res: &mut VecZnxBig<B>, col_res: usize, a: &VecZnxBig<B>, col_a: usize);

    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_small(
        &self,
        res: &mut VecZnxBig<B>,
        col_res: usize,
        a: &VecZnx,
        col_a: usize,
        b: &VecZnxBig<B>,
        col_b: usize,
    );

    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_small_inplace(&self, res: &mut VecZnxBig<B>, col_res: usize, a: &VecZnx, col_a: usize);

    /// Subtracts `a` to `b` and stores the result on `c`.
    fn vec_znx_big_sub(
        &self,
        res: &mut VecZnxBig<B>,
        col_res: usize,
        a: &VecZnxBig<B>,
        col_a: usize,
        b: &VecZnxBig<B>,
        col_b: usize,
    );

    /// Subtracts `a` to `b` and stores the result on `b`.
    fn vec_znx_big_sub_ab_inplace(&self, res: &mut VecZnxBig<B>, col_res: usize, a: &VecZnxBig<B>, col_a: usize);

    /// Subtracts `b` to `a` and stores the result on `b`.
    fn vec_znx_big_sub_ba_inplace(&self, res: &mut VecZnxBig<B>, col_res: usize, a: &VecZnxBig<B>, col_a: usize);

    /// Subtracts `b` to `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_a(
        &self,
        res: &mut VecZnxBig<B>,
        col_res: usize,
        a: &VecZnx,
        col_a: usize,
        b: &VecZnxBig<B>,
        col_b: usize,
    );

    /// Subtracts `a` to `b` and stores the result on `b`.
    fn vec_znx_big_sub_small_a_inplace(&self, res: &mut VecZnxBig<B>, col_res: usize, a: &VecZnx, col_a: usize);

    /// Subtracts `b` to `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_b(
        &self,
        res: &mut VecZnxBig<B>,
        col_res: usize,
        a: &VecZnxBig<B>,
        col_a: usize,
        b: &VecZnx,
        col_b: usize,
    );

    /// Subtracts `b` to `a` and stores the result on `b`.
    fn vec_znx_big_sub_small_b_inplace(&self, res: &mut VecZnxBig<B>, col_res: usize, a: &VecZnx, col_a: usize);

    /// Returns the minimum number of bytes to apply [VecZnxBigOps::vec_znx_big_normalize].
    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize;

    /// Normalizes `a` and stores the result on `b`.
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: normalization basis.
    /// * `tmp_bytes`: scratch space of size at least [VecZnxBigOps::vec_znx_big_normalize].
    fn vec_znx_big_normalize(
        &self,
        log_base2k: usize,
        res: &mut VecZnx,
        col_res: usize,
        a: &VecZnxBig<B>,
        col_a: usize,
        tmp_bytes: &mut [u8],
    );

    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `b`.
    fn vec_znx_big_automorphism(&self, k: i64, res: &mut VecZnxBig<B>, col_res: usize, a: &VecZnxBig<B>, col_a: usize);

    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `a`.
    fn vec_znx_big_automorphism_inplace(&self, k: i64, a: &mut VecZnxBig<B>, col_a: usize);
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

    fn vec_znx_big_add(
        &self,
        res: &mut VecZnxBig<FFT64>,
        col_res: usize,
        a: &VecZnxBig<FFT64>,
        col_a: usize,
        b: &VecZnxBig<FFT64>,
        col_b: usize,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
                res.at_mut_ptr(col_res, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(col_a, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(col_b, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }

    fn vec_znx_big_add_inplace(&self, res: &mut VecZnxBig<FFT64>, col_res: usize, a: &VecZnxBig<FFT64>, col_a: usize) {
        unsafe {
            let res_ptr: *mut VecZnxBig<FFT64> = res as *mut VecZnxBig<FFT64>;
            Self::vec_znx_big_add(self, &mut *res_ptr, col_res, a, col_a, &*res_ptr, col_res);
        }
    }

    fn vec_znx_big_sub(
        &self,
        res: &mut VecZnxBig<FFT64>,
        col_res: usize,
        a: &VecZnxBig<FFT64>,
        col_a: usize,
        b: &VecZnxBig<FFT64>,
        col_b: usize,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                res.at_mut_ptr(col_res, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(col_a, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(col_b, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }

    fn vec_znx_big_sub_ab_inplace(&self, res: &mut VecZnxBig<FFT64>, col_res: usize, a: &VecZnxBig<FFT64>, col_a: usize) {
        unsafe {
            let res_ptr: *mut VecZnxBig<FFT64> = res as *mut VecZnxBig<FFT64>;
            Self::vec_znx_big_sub(self, &mut *res_ptr, col_res, a, col_a, &*res_ptr, col_res);
        }
    }

    fn vec_znx_big_sub_ba_inplace(&self, res: &mut VecZnxBig<FFT64>, col_res: usize, a: &VecZnxBig<FFT64>, col_a: usize) {
        unsafe {
            let res_ptr: *mut VecZnxBig<FFT64> = res as *mut VecZnxBig<FFT64>;
            Self::vec_znx_big_sub(self, &mut *res_ptr, col_res, &*res_ptr, col_res, a, col_a);
        }
    }

    fn vec_znx_big_sub_small_b(
        &self,
        res: &mut VecZnxBig<FFT64>,
        col_res: usize,
        a: &VecZnxBig<FFT64>,
        col_a: usize,
        b: &VecZnx,
        col_b: usize,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                res.at_mut_ptr(col_res, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(col_a, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(col_b, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }

    fn vec_znx_big_sub_small_b_inplace(&self, res: &mut VecZnxBig<FFT64>, col_res: usize, a: &VecZnx, col_a: usize) {
        unsafe {
            let res_ptr: *mut VecZnxBig<FFT64> = res as *mut VecZnxBig<FFT64>;
            Self::vec_znx_big_sub_small_b(self, &mut *res_ptr, col_res, &*res_ptr, col_res, a, col_a);
        }
    }

    fn vec_znx_big_sub_small_a(
        &self,
        res: &mut VecZnxBig<FFT64>,
        col_res: usize,
        a: &VecZnx,
        col_a: usize,
        b: &VecZnxBig<FFT64>,
        col_b: usize,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                res.at_mut_ptr(col_res, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(col_a, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(col_b, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }

    fn vec_znx_big_sub_small_a_inplace(&self, res: &mut VecZnxBig<FFT64>, col_res: usize, a: &VecZnx, col_a: usize) {
        unsafe {
            let res_ptr: *mut VecZnxBig<FFT64> = res as *mut VecZnxBig<FFT64>;
            Self::vec_znx_big_sub_small_a(self, &mut *res_ptr, col_res, a, col_a, &*res_ptr, col_res);
        }
    }

    fn vec_znx_big_add_small(
        &self,
        res: &mut VecZnxBig<FFT64>,
        col_res: usize,
        a: &VecZnx,
        col_a: usize,
        b: &VecZnxBig<FFT64>,
        col_b: usize,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
                res.at_mut_ptr(col_res, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(col_a, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(col_b, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }

    fn vec_znx_big_add_small_inplace(&self, res: &mut VecZnxBig<FFT64>, col_res: usize, a: &VecZnx, a_col: usize) {
        unsafe {
            let res_ptr: *mut VecZnxBig<FFT64> = res as *mut VecZnxBig<FFT64>;
            Self::vec_znx_big_add_small(self, &mut *res_ptr, col_res, a, a_col, &*res_ptr, col_res);
        }
    }

    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize {
        Self::vec_znx_normalize_tmp_bytes(self)
    }

    fn vec_znx_big_normalize(
        &self,
        log_base2k: usize,
        res: &mut VecZnx,
        col_res: usize,
        a: &VecZnxBig<FFT64>,
        col_a: usize,
        tmp_bytes: &mut [u8],
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert!(tmp_bytes.len() >= Self::vec_znx_normalize_tmp_bytes(&self));
            assert_alignement(tmp_bytes.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_normalize_base2k(
                self.ptr,
                log_base2k as u64,
                res.at_mut_ptr(col_res, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(col_a, 0),
                a.size() as u64,
                a.sl() as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vec_znx_big_automorphism(&self, k: i64, res: &mut VecZnxBig<FFT64>, col_res: usize, a: &VecZnxBig<FFT64>, col_a: usize) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.ptr,
                k,
                res.at_mut_ptr(col_res, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(col_a, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }

    fn vec_znx_big_automorphism_inplace(&self, k: i64, a: &mut VecZnxBig<FFT64>, col_a: usize) {
        unsafe {
            let a_ptr: *mut VecZnxBig<FFT64> = a as *mut VecZnxBig<FFT64>;
            Self::vec_znx_big_automorphism(self, k, &mut *a_ptr, col_a, &*a_ptr, col_a);
        }
    }
}
