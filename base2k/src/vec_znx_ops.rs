use crate::ffi::vec_znx;
use crate::znx_base::{ZnxAlloc, ZnxInfos, ZnxLayout, ZnxSliceSize, switch_degree};
use crate::{Backend, Module, VEC_ZNX_ROWS, VecZnx, assert_alignement};
pub trait VecZnxOps {
    /// Allocates a new [VecZnx].
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of polynomials.
    /// * `size`: the number small polynomials per column.
    fn new_vec_znx(&self, cols: usize, size: usize) -> VecZnx;

    /// Instantiates a new [VecZnx] from a slice of bytes.
    /// The returned [VecZnx] takes ownership of the slice of bytes.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of polynomials.
    /// * `size`: the number small polynomials per column.
    ///
    /// # Panic
    /// Requires the slice of bytes to be equal to [VecZnxOps::bytes_of_vec_znx].
    fn new_vec_znx_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnx;

    /// Instantiates a new [VecZnx] from a slice of bytes.
    /// The returned [VecZnx] does take ownership of the slice of bytes.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of polynomials.
    /// * `size`: the number small polynomials per column.
    ///
    /// # Panic
    /// Requires the slice of bytes to be equal to [VecZnxOps::bytes_of_vec_znx].
    fn new_vec_znx_from_bytes_borrow(&self, cols: usize, size: usize, tmp_bytes: &mut [u8]) -> VecZnx;

    /// Returns the number of bytes necessary to allocate
    /// a new [VecZnx] through [VecZnxOps::new_vec_znx_from_bytes]
    /// or [VecZnxOps::new_vec_znx_from_bytes_borrow].
    fn bytes_of_vec_znx(&self, cols: usize, size: usize) -> usize;

    /// Returns the minimum number of bytes necessary for normalization.
    fn vec_znx_normalize_tmp_bytes(&self) -> usize;

    /// Normalizes the selected column of `a` and stores the result into the selected column of `res`.
    fn vec_znx_normalize(
        &self,
        log_base2k: usize,
        res: &mut VecZnx,
        col_res: usize,
        a: &VecZnx,
        col_a: usize,
        tmp_bytes: &mut [u8],
    );

    /// Normalizes the selected column of `a`.
    fn vec_znx_normalize_inplace(&self, log_base2k: usize, a: &mut VecZnx, col_a: usize, tmp_bytes: &mut [u8]);

    /// Adds the selected column of `a` to the selected column of `b` and write the result on the selected column of `c`.
    fn vec_znx_add(&self, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize, b: &VecZnx, col_b: usize);

    /// Adds the selected column of `a` to the selected column of `b` and write the result on the selected column of `res`.
    fn vec_znx_add_inplace(&self, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize);

    /// Subtracts the selected column of `b` to the selected column of `a` and write the result on the selected column of `res`.
    fn vec_znx_sub(&self, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize, b: &VecZnx, col_b: usize);

    /// Subtracts the selected column of `a` to the selected column of `res`.
    fn vec_znx_sub_ab_inplace(&self, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize);

    /// Subtracts the selected column of `a` to the selected column of `res` and negates the selected column of `res`.
    fn vec_znx_sub_ba_inplace(&self, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize);

    // Negates the selected column of `a` and stores the result on the selected column of `res`.
    fn vec_znx_negate(&self, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize);

    /// Negates the selected column of `a`.
    fn vec_znx_negate_inplace(&self, a: &mut VecZnx, col_a: usize);

    /// Multiplies the selected column of `a` by X^k and stores the result on the selected column of `res`.
    fn vec_znx_rotate(&self, k: i64, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize);

    /// Multiplies the selected column of `a` by X^k.
    fn vec_znx_rotate_inplace(&self, k: i64, a: &mut VecZnx, col_a: usize);

    /// Applies the automorphism X^i -> X^ik on the selected column of `a` and stores the result on the selected column of `res`.
    fn vec_znx_automorphism(&self, k: i64, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize);

    /// Applies the automorphism X^i -> X^ik on the selected column of `a`.
    fn vec_znx_automorphism_inplace(&self, k: i64, a: &mut VecZnx, col_a: usize);

    /// Splits the selected columns of `b` into subrings and copies them them into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of b have the same ring degree
    /// and that b.n() * b.len() <= a.n()
    fn vec_znx_split(&self, res: &mut Vec<VecZnx>, col_res: usize, a: &VecZnx, col_a: usize, buf: &mut VecZnx);

    /// Merges the subrings of the selected column of `a` into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of a have the same ring degree
    /// and that a.n() * a.len() <= b.n()
    fn vec_znx_merge(&self, res: &mut VecZnx, col_res: usize, a: &Vec<VecZnx>, col_a: usize);
}

impl<B: Backend> VecZnxOps for Module<B> {
    fn new_vec_znx(&self, cols: usize, size: usize) -> VecZnx {
        VecZnx::new(self, VEC_ZNX_ROWS, cols, size)
    }

    fn bytes_of_vec_znx(&self, cols: usize, size: usize) -> usize {
        VecZnx::bytes_of(self, VEC_ZNX_ROWS, cols, size)
    }

    fn new_vec_znx_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnx {
        VecZnx::from_bytes(self, VEC_ZNX_ROWS, cols, size, bytes)
    }

    fn new_vec_znx_from_bytes_borrow(&self, cols: usize, size: usize, tmp_bytes: &mut [u8]) -> VecZnx {
        VecZnx::from_bytes_borrow(self, VEC_ZNX_ROWS, cols, size, tmp_bytes)
    }

    fn vec_znx_normalize_tmp_bytes(&self) -> usize {
        unsafe { vec_znx::vec_znx_normalize_base2k_tmp_bytes(self.ptr) as usize }
    }

    fn vec_znx_normalize(
        &self,
        log_base2k: usize,
        res: &mut VecZnx,
        col_res: usize,
        a: &VecZnx,
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

    fn vec_znx_normalize_inplace(&self, log_base2k: usize, a: &mut VecZnx, col_a: usize, tmp_bytes: &mut [u8]) {
        unsafe {
            let a_ptr: *mut VecZnx = a as *mut VecZnx;
            Self::vec_znx_normalize(
                self,
                log_base2k,
                &mut *a_ptr,
                col_a,
                &*a_ptr,
                col_a,
                tmp_bytes,
            );
        }
    }

    fn vec_znx_add(&self, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize, b: &VecZnx, col_b: usize) {
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

    fn vec_znx_add_inplace(&self, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize) {
        unsafe {
            let res_ptr: *mut VecZnx = res as *mut VecZnx;
            Self::vec_znx_add(self, &mut *res_ptr, col_res, a, col_a, &*res_ptr, col_res);
        }
    }

    fn vec_znx_sub(&self, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize, b: &VecZnx, col_b: usize) {
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

    fn vec_znx_sub_ab_inplace(&self, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize) {
        unsafe {
            let res_ptr: *mut VecZnx = res as *mut VecZnx;
            Self::vec_znx_sub(self, &mut *res_ptr, col_res, a, col_a, &*res_ptr, col_res);
        }
    }

    fn vec_znx_sub_ba_inplace(&self, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize) {
        unsafe {
            let res_ptr: *mut VecZnx = res as *mut VecZnx;
            Self::vec_znx_sub(self, &mut *res_ptr, col_res, &*res_ptr, col_res, a, col_a);
        }
    }

    fn vec_znx_negate(&self, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_negate(
                self.ptr,
                res.at_mut_ptr(col_res, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(col_a, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }

    fn vec_znx_negate_inplace(&self, a: &mut VecZnx, col_a: usize) {
        unsafe {
            let a_ptr: *mut VecZnx = a as *mut VecZnx;
            Self::vec_znx_negate(self, &mut *a_ptr, col_a, &*a_ptr, col_a);
        }
    }

    fn vec_znx_rotate(&self, k: i64, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_rotate(
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

    fn vec_znx_rotate_inplace(&self, k: i64, a: &mut VecZnx, col_a: usize) {
        unsafe {
            let a_ptr: *mut VecZnx = a as *mut VecZnx;
            Self::vec_znx_rotate(self, k, &mut *a_ptr, col_a, &*a_ptr, col_a);
        }
    }

    fn vec_znx_automorphism(&self, k: i64, res: &mut VecZnx, col_res: usize, a: &VecZnx, col_a: usize) {
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

    fn vec_znx_automorphism_inplace(&self, k: i64, a: &mut VecZnx, col_a: usize) {
        unsafe {
            let a_ptr: *mut VecZnx = a as *mut VecZnx;
            Self::vec_znx_automorphism(self, k, &mut *a_ptr, col_a, &*a_ptr, col_a);
        }
    }

    fn vec_znx_split(&self, res: &mut Vec<VecZnx>, col_res: usize, a: &VecZnx, col_a: usize, buf: &mut VecZnx) {
        let (n_in, n_out) = (a.n(), res[0].n());

        debug_assert!(
            n_out < n_in,
            "invalid a: output ring degree should be smaller"
        );
        res[1..].iter().for_each(|bi| {
            debug_assert_eq!(
                bi.n(),
                n_out,
                "invalid input a: all VecZnx must have the same degree"
            )
        });

        res.iter_mut().enumerate().for_each(|(i, bi)| {
            if i == 0 {
                switch_degree(bi, col_res, a, col_a);
                self.vec_znx_rotate(-1, buf, 0, a, col_a);
            } else {
                switch_degree(bi, col_res, buf, col_a);
                self.vec_znx_rotate_inplace(-1, buf, col_a);
            }
        })
    }

    fn vec_znx_merge(&self, res: &mut VecZnx, col_res: usize, a: &Vec<VecZnx>, col_a: usize) {
        let (n_in, n_out) = (res.n(), a[0].n());

        debug_assert!(
            n_out < n_in,
            "invalid a: output ring degree should be smaller"
        );
        a[1..].iter().for_each(|ai| {
            debug_assert_eq!(
                ai.n(),
                n_out,
                "invalid input a: all VecZnx must have the same degree"
            )
        });

        a.iter().enumerate().for_each(|(_, ai)| {
            switch_degree(res, col_res, ai, col_a);
            self.vec_znx_rotate_inplace(-1, res, col_res);
        });

        self.vec_znx_rotate_inplace(a.len() as i64, res, col_res);
    }
}
