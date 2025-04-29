use crate::ffi::module::MODULE;
use crate::ffi::vec_znx;
use crate::{Backend, Module, VecZnx, ZnxBase, ZnxBasics, ZnxInfos, ZnxLayout, switch_degree, znx_post_process_ternary_op};
use std::cmp::min;
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
    fn new_vec_znx_from_bytes(&self, cols: usize, size: usize, bytes: &mut [u8]) -> VecZnx;

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
    fn vec_znx_normalize_tmp_bytes(&self, cols: usize) -> usize;

    /// Adds `a` to `b` and write the result on `c`.
    fn vec_znx_add(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx);

    /// Adds `a` to `b` and write the result on `b`.
    fn vec_znx_add_inplace(&self, b: &mut VecZnx, a: &VecZnx);

    /// Subtracts `b` to `a` and write the result on `c`.
    fn vec_znx_sub(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx);

    /// Subtracts `a` to `b` and write the result on `b`.
    fn vec_znx_sub_ab_inplace(&self, b: &mut VecZnx, a: &VecZnx);

    /// Subtracts `b` to `a` and write the result on `b`.
    fn vec_znx_sub_ba_inplace(&self, b: &mut VecZnx, a: &VecZnx);

    // Negates `a` and stores the result on `b`.
    fn vec_znx_negate(&self, b: &mut VecZnx, a: &VecZnx);

    /// Negages `a` and stores the result on `a`.
    fn vec_znx_negate_inplace(&self, a: &mut VecZnx);

    /// Multiplies `a` by X^k and stores the result on `b`.
    fn vec_znx_rotate(&self, k: i64, b: &mut VecZnx, a: &VecZnx);

    /// Multiplies `a` by X^k and stores the result on `a`.
    fn vec_znx_rotate_inplace(&self, k: i64, a: &mut VecZnx);

    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `b`.
    fn vec_znx_automorphism(&self, k: i64, b: &mut VecZnx, a: &VecZnx);

    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `a`.
    fn vec_znx_automorphism_inplace(&self, k: i64, a: &mut VecZnx);

    /// Splits b into subrings and copies them them into a.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of b have the same ring degree
    /// and that b.n() * b.len() <= a.n()
    fn vec_znx_split(&self, b: &mut Vec<VecZnx>, a: &VecZnx, buf: &mut VecZnx);

    /// Merges the subrings a into b.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of a have the same ring degree
    /// and that a.n() * a.len() <= b.n()
    fn vec_znx_merge(&self, b: &mut VecZnx, a: &Vec<VecZnx>);
}

impl<B: Backend> VecZnxOps for Module<B> {
    fn new_vec_znx(&self, cols: usize, size: usize) -> VecZnx {
        VecZnx::new(self, cols, size)
    }

    fn bytes_of_vec_znx(&self, cols: usize, size: usize) -> usize {
        VecZnx::bytes_of(self, cols, size)
    }

    fn new_vec_znx_from_bytes(&self, cols: usize, size: usize, bytes: &mut [u8]) -> VecZnx {
        VecZnx::from_bytes(self, cols, size, bytes)
    }

    fn new_vec_znx_from_bytes_borrow(&self, cols: usize, size: usize, tmp_bytes: &mut [u8]) -> VecZnx {
        VecZnx::from_bytes_borrow(self, cols, size, tmp_bytes)
    }

    fn vec_znx_normalize_tmp_bytes(&self, cols: usize) -> usize {
        unsafe { vec_znx::vec_znx_normalize_base2k_tmp_bytes(self.ptr) as usize * cols }
    }

    fn vec_znx_add(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx) {
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
        vec_znx_apply_binary_op::<B, false>(self, c, a, b, op);
    }

    fn vec_znx_add_inplace(&self, b: &mut VecZnx, a: &VecZnx) {
        unsafe {
            let b_ptr: *mut VecZnx = b as *mut VecZnx;
            Self::vec_znx_add(self, &mut *b_ptr, a, &*b_ptr);
        }
    }

    fn vec_znx_sub(&self, c: &mut VecZnx, a: &VecZnx, b: &VecZnx) {
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
        vec_znx_apply_binary_op::<B, true>(self, c, a, b, op);
    }

    fn vec_znx_sub_ab_inplace(&self, b: &mut VecZnx, a: &VecZnx) {
        unsafe {
            let b_ptr: *mut VecZnx = b as *mut VecZnx;
            Self::vec_znx_sub(self, &mut *b_ptr, a, &*b_ptr);
        }
    }

    fn vec_znx_sub_ba_inplace(&self, b: &mut VecZnx, a: &VecZnx) {
        unsafe {
            let b_ptr: *mut VecZnx = b as *mut VecZnx;
            Self::vec_znx_sub(self, &mut *b_ptr, &*b_ptr, a);
        }
    }

    fn vec_znx_negate(&self, b: &mut VecZnx, a: &VecZnx) {
        let op = ffi_binary_op_factory_type_0(
            self.ptr,
            b.size(),
            b.sl(),
            a.size(),
            a.sl(),
            vec_znx::vec_znx_negate,
        );
        vec_znx_apply_unary_op::<B>(self, b, a, op);
    }

    fn vec_znx_negate_inplace(&self, a: &mut VecZnx) {
        unsafe {
            let a_ptr: *mut VecZnx = a as *mut VecZnx;
            Self::vec_znx_negate(self, &mut *a_ptr, &*a_ptr);
        }
    }

    fn vec_znx_rotate(&self, k: i64, b: &mut VecZnx, a: &VecZnx) {
        let op = ffi_binary_op_factory_type_1(
            self.ptr,
            k,
            b.size(),
            b.sl(),
            a.size(),
            a.sl(),
            vec_znx::vec_znx_rotate,
        );
        vec_znx_apply_unary_op::<B>(self, b, a, op);
    }

    fn vec_znx_rotate_inplace(&self, k: i64, a: &mut VecZnx) {
        unsafe {
            let a_ptr: *mut VecZnx = a as *mut VecZnx;
            Self::vec_znx_rotate(self, k, &mut *a_ptr, &*a_ptr);
        }
    }

    fn vec_znx_automorphism(&self, k: i64, b: &mut VecZnx, a: &VecZnx) {
        let op = ffi_binary_op_factory_type_1(
            self.ptr,
            k,
            b.size(),
            b.sl(),
            a.size(),
            a.sl(),
            vec_znx::vec_znx_automorphism,
        );
        vec_znx_apply_unary_op::<B>(self, b, a, op);
    }

    fn vec_znx_automorphism_inplace(&self, k: i64, a: &mut VecZnx) {
        unsafe {
            let a_ptr: *mut VecZnx = a as *mut VecZnx;
            Self::vec_znx_automorphism(self, k, &mut *a_ptr, &*a_ptr);
        }
    }

    fn vec_znx_split(&self, b: &mut Vec<VecZnx>, a: &VecZnx, buf: &mut VecZnx) {
        let (n_in, n_out) = (a.n(), b[0].n());

        debug_assert!(
            n_out < n_in,
            "invalid a: output ring degree should be smaller"
        );
        b[1..].iter().for_each(|bi| {
            debug_assert_eq!(
                bi.n(),
                n_out,
                "invalid input a: all VecZnx must have the same degree"
            )
        });

        b.iter_mut().enumerate().for_each(|(i, bi)| {
            if i == 0 {
                switch_degree(bi, a);
                self.vec_znx_rotate(-1, buf, a);
            } else {
                switch_degree(bi, buf);
                self.vec_znx_rotate_inplace(-1, buf);
            }
        })
    }

    fn vec_znx_merge(&self, b: &mut VecZnx, a: &Vec<VecZnx>) {
        let (n_in, n_out) = (b.n(), a[0].n());

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
            switch_degree(b, ai);
            self.vec_znx_rotate_inplace(-1, b);
        });

        self.vec_znx_rotate_inplace(a.len() as i64, b);
    }
}

fn ffi_ternary_op_factory(
    module_ptr: *const MODULE,
    c_size: usize,
    c_sl: usize,
    a_size: usize,
    a_sl: usize,
    b_size: usize,
    b_sl: usize,
    op_fn: unsafe extern "C" fn(*const MODULE, *mut i64, u64, u64, *const i64, u64, u64, *const i64, u64, u64),
) -> impl Fn(&mut [i64], &[i64], &[i64]) {
    move |cv: &mut [i64], av: &[i64], bv: &[i64]| unsafe {
        op_fn(
            module_ptr,
            cv.as_mut_ptr(),
            c_size as u64,
            c_sl as u64,
            av.as_ptr(),
            a_size as u64,
            a_sl as u64,
            bv.as_ptr(),
            b_size as u64,
            b_sl as u64,
        )
    }
}

fn ffi_binary_op_factory_type_0(
    module_ptr: *const MODULE,
    b_size: usize,
    b_sl: usize,
    a_size: usize,
    a_sl: usize,
    op_fn: unsafe extern "C" fn(*const MODULE, *mut i64, u64, u64, *const i64, u64, u64),
) -> impl Fn(&mut [i64], &[i64]) {
    move |bv: &mut [i64], av: &[i64]| unsafe {
        op_fn(
            module_ptr,
            bv.as_mut_ptr(),
            b_size as u64,
            b_sl as u64,
            av.as_ptr(),
            a_size as u64,
            a_sl as u64,
        )
    }
}

fn ffi_binary_op_factory_type_1(
    module_ptr: *const MODULE,
    k: i64,
    b_size: usize,
    b_sl: usize,
    a_size: usize,
    a_sl: usize,
    op_fn: unsafe extern "C" fn(*const MODULE, i64, *mut i64, u64, u64, *const i64, u64, u64),
) -> impl Fn(&mut [i64], &[i64]) {
    move |bv: &mut [i64], av: &[i64]| unsafe {
        op_fn(
            module_ptr,
            k,
            bv.as_mut_ptr(),
            b_size as u64,
            b_sl as u64,
            av.as_ptr(),
            a_size as u64,
            a_sl as u64,
        )
    }
}

#[inline(always)]
pub fn vec_znx_apply_binary_op<B: Backend, const NEGATE: bool>(
    module: &Module<B>,
    c: &mut VecZnx,
    a: &VecZnx,
    b: &VecZnx,
    op: impl Fn(&mut [i64], &[i64], &[i64]),
) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), module.n());
        assert_eq!(b.n(), module.n());
        assert_eq!(c.n(), module.n());
        assert_ne!(a.as_ptr(), b.as_ptr());
    }
    let a_cols: usize = a.cols();
    let b_cols: usize = b.cols();
    let c_cols: usize = c.cols();
    let min_ab_cols: usize = min(a_cols, b_cols);
    let min_cols: usize = min(c_cols, min_ab_cols);
    // Applies over shared cols between (a, b, c)
    (0..min_cols).for_each(|i| op(c.at_poly_mut(i, 0), a.at_poly(i, 0), b.at_poly(i, 0)));
    // Copies/Negates/Zeroes the remaining cols if op is not inplace.
    if c.as_ptr() != a.as_ptr() && c.as_ptr() != b.as_ptr() {
        znx_post_process_ternary_op::<VecZnx, NEGATE>(c, a, b);
    }
}

#[inline(always)]
pub fn vec_znx_apply_unary_op<B: Backend>(module: &Module<B>, b: &mut VecZnx, a: &VecZnx, op: impl Fn(&mut [i64], &[i64])) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), module.n());
        assert_eq!(b.n(), module.n());
    }
    let a_cols: usize = a.cols();
    let b_cols: usize = b.cols();
    let min_cols: usize = min(a_cols, b_cols);
    // Applies over the shared cols between (a, b)
    (0..min_cols).for_each(|i| op(b.at_poly_mut(i, 0), a.at_poly(i, 0)));
    // Zeroes the remaining cols of b.
    (min_cols..b_cols).for_each(|i| (0..b.size()).for_each(|j| b.zero_at(i, j)));
}

#[cfg(test)]
mod tests {
    use crate::{
        Backend, FFT64, Module, Sampling, VecZnx, VecZnxOps, ZnxBasics, ZnxInfos, ZnxLayout, ffi::vec_znx,
        znx_post_process_ternary_op,
    };
    use itertools::izip;
    use sampling::source::Source;
    use std::cmp::min;

    #[test]
    fn vec_znx_add() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let op = |cv: &mut [i64], av: &[i64], bv: &[i64]| {
            izip!(cv.iter_mut(), bv.iter(), av.iter()).for_each(|(ci, bi, ai)| *ci = *bi + *ai);
        };
        test_binary_op::<false, _>(
            &module,
            &|c: &mut VecZnx, a: &VecZnx, b: &VecZnx| module.vec_znx_add(c, a, b),
            op,
        );
    }

    #[test]
    fn vec_znx_add_inplace() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let op = |bv: &mut [i64], av: &[i64]| {
            izip!(bv.iter_mut(), av.iter()).for_each(|(bi, ai)| *bi = *bi + *ai);
        };
        test_binary_op_inplace::<false, _>(
            &module,
            &|b: &mut VecZnx, a: &VecZnx| module.vec_znx_add_inplace(b, a),
            op,
        );
    }

    #[test]
    fn vec_znx_sub() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let op = |cv: &mut [i64], av: &[i64], bv: &[i64]| {
            izip!(cv.iter_mut(), bv.iter(), av.iter()).for_each(|(ci, bi, ai)| *ci = *bi - *ai);
        };
        test_binary_op::<true, _>(
            &module,
            &|c: &mut VecZnx, a: &VecZnx, b: &VecZnx| module.vec_znx_sub(c, a, b),
            op,
        );
    }

    #[test]
    fn vec_znx_sub_ab_inplace() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let op = |bv: &mut [i64], av: &[i64]| {
            izip!(bv.iter_mut(), av.iter()).for_each(|(bi, ai)| *bi = *ai - *bi);
        };
        test_binary_op_inplace::<true, _>(
            &module,
            &|b: &mut VecZnx, a: &VecZnx| module.vec_znx_sub_ab_inplace(b, a),
            op,
        );
    }

    #[test]
    fn vec_znx_sub_ba_inplace() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let op = |bv: &mut [i64], av: &[i64]| {
            izip!(bv.iter_mut(), av.iter()).for_each(|(bi, ai)| *bi = *bi - *ai);
        };
        test_binary_op_inplace::<false, _>(
            &module,
            &|b: &mut VecZnx, a: &VecZnx| module.vec_znx_sub_ba_inplace(b, a),
            op,
        );
    }

    #[test]
    fn vec_znx_negate() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let op = |b: &mut [i64], a: &[i64]| {
            izip!(b.iter_mut(), a.iter()).for_each(|(bi, ai)| *bi = -*ai);
        };
        test_unary_op(
            &module,
            |b: &mut VecZnx, a: &VecZnx| module.vec_znx_negate(b, a),
            op,
        )
    }

    #[test]
    fn vec_znx_negate_inplace() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let op = |a: &mut [i64]| a.iter_mut().for_each(|xi| *xi = -*xi);
        test_unary_op_inplace(
            &module,
            |a: &mut VecZnx| module.vec_znx_negate_inplace(a),
            op,
        )
    }

    #[test]
    fn vec_znx_rotate() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let k: i64 = 53;
        let op = |b: &mut [i64], a: &[i64]| {
            assert_eq!(b.len(), a.len());
            b.copy_from_slice(a);

            let mut k_mod2n: i64 = k % (2 * n as i64);
            if k_mod2n < 0 {
                k_mod2n += 2 * n as i64;
            }
            let sign: i64 = (k_mod2n.abs() / (n as i64)) & 1;
            let k_modn: i64 = k_mod2n % (n as i64);

            b.rotate_right(k_modn as usize);
            b[0..k_modn as usize].iter_mut().for_each(|x| *x = -*x);

            if sign == 1 {
                b.iter_mut().for_each(|x| *x = -*x);
            }
        };
        test_unary_op(
            &module,
            |b: &mut VecZnx, a: &VecZnx| module.vec_znx_rotate(k, b, a),
            op,
        )
    }

    #[test]
    fn vec_znx_rotate_inplace() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let k: i64 = 53;
        let rot = |a: &mut [i64]| {
            let mut k_mod2n: i64 = k % (2 * n as i64);
            if k_mod2n < 0 {
                k_mod2n += 2 * n as i64;
            }
            let sign: i64 = (k_mod2n.abs() / (n as i64)) & 1;
            let k_modn: i64 = k_mod2n % (n as i64);

            a.rotate_right(k_modn as usize);
            a[0..k_modn as usize].iter_mut().for_each(|x| *x = -*x);

            if sign == 1 {
                a.iter_mut().for_each(|x| *x = -*x);
            }
        };
        test_unary_op_inplace(
            &module,
            |a: &mut VecZnx| module.vec_znx_rotate_inplace(k, a),
            rot,
        )
    }

    #[test]
    fn vec_znx_automorphism() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let k: i64 = -5;
        let op = |b: &mut [i64], a: &[i64]| {
            assert_eq!(b.len(), a.len());
            unsafe {
                vec_znx::vec_znx_automorphism(
                    module.ptr,
                    k,
                    b.as_mut_ptr(),
                    1u64,
                    n as u64,
                    a.as_ptr(),
                    1u64,
                    n as u64,
                );
            }
        };
        test_unary_op(
            &module,
            |b: &mut VecZnx, a: &VecZnx| module.vec_znx_automorphism(k, b, a),
            op,
        )
    }

    #[test]
    fn vec_znx_automorphism_inplace() {
        let n: usize = 8;
        let module: Module<FFT64> = Module::<FFT64>::new(n);
        let k: i64 = -5;
        let op = |a: &mut [i64]| unsafe {
            vec_znx::vec_znx_automorphism(
                module.ptr,
                k,
                a.as_mut_ptr(),
                1u64,
                n as u64,
                a.as_ptr(),
                1u64,
                n as u64,
            );
        };
        test_unary_op_inplace(
            &module,
            |a: &mut VecZnx| module.vec_znx_automorphism_inplace(k, a),
            op,
        )
    }

    fn test_binary_op<const NEGATE: bool, B: Backend>(
        module: &Module<B>,
        func_have: impl Fn(&mut VecZnx, &VecZnx, &VecZnx),
        func_want: impl Fn(&mut [i64], &[i64], &[i64]),
    ) {
        let a_size: usize = 3;
        let b_size: usize = 4;
        let c_size: usize = 5;
        let mut source: Source = Source::new([0u8; 32]);

        [1usize, 2, 3].iter().for_each(|a_cols| {
            [1usize, 2, 3].iter().for_each(|b_cols| {
                [1usize, 2, 3].iter().for_each(|c_cols| {
                    let min_ab_cols: usize = min(*a_cols, *b_cols);
                    let min_cols: usize = min(*c_cols, min_ab_cols);
                    let min_size: usize = min(c_size, min(a_size, b_size));

                    // Allocats a and populates with random values.
                    let mut a: VecZnx = module.new_vec_znx(*a_cols, a_size);
                    (0..*a_cols).for_each(|i| {
                        module.fill_uniform(3, &mut a, i, a_size, &mut source);
                    });

                    // Allocats b and populates with random values.
                    let mut b: VecZnx = module.new_vec_znx(*b_cols, b_size);
                    (0..*b_cols).for_each(|i| {
                        module.fill_uniform(3, &mut b, i, b_size, &mut source);
                    });

                    // Allocats c and populates with random values.
                    let mut c_have: VecZnx = module.new_vec_znx(*c_cols, c_size);
                    (0..c_have.cols()).for_each(|i| {
                        module.fill_uniform(3, &mut c_have, i, c_size, &mut source);
                    });

                    // Applies the function to test
                    func_have(&mut c_have, &a, &b);

                    let mut c_want: VecZnx = module.new_vec_znx(*c_cols, c_size);

                    // Applies the reference function and expected behavior.
                    // Adds with the minimum matching columns
                    (0..min_cols).for_each(|i| {
                        // Adds with th eminimum matching size
                        (0..min_size).for_each(|j| {
                            func_want(c_want.at_poly_mut(i, j), b.at_poly(i, j), a.at_poly(i, j));
                        });

                        if a_size > b_size {
                            // Copies remaining size of lh if lh.size() > rh.size()
                            (min_size..a_size).for_each(|j| {
                                izip!(c_want.at_poly_mut(i, j).iter_mut(), a.at_poly(i, j).iter()).for_each(|(ci, ai)| *ci = *ai);
                                if NEGATE {
                                    c_want.at_poly_mut(i, j).iter_mut().for_each(|x| *x = -*x);
                                }
                            });
                        } else {
                            // Copies the remaining size of rh if the are greater
                            (min_size..b_size).for_each(|j| {
                                izip!(c_want.at_poly_mut(i, j).iter_mut(), b.at_poly(i, j).iter()).for_each(|(ci, bi)| *ci = *bi);
                                if NEGATE {
                                    c_want.at_poly_mut(i, j).iter_mut().for_each(|x| *x = -*x);
                                }
                            });
                        }
                    });

                    znx_post_process_ternary_op::<_, NEGATE>(&mut c_want, &a, &b);

                    assert_eq!(c_have.raw(), c_want.raw());
                });
            });
        });
    }

    fn test_binary_op_inplace<const NEGATE: bool, B: Backend>(
        module: &Module<B>,
        func_have: impl Fn(&mut VecZnx, &VecZnx),
        func_want: impl Fn(&mut [i64], &[i64]),
    ) {
        let a_size: usize = 3;
        let b_size: usize = 5;
        let mut source = Source::new([0u8; 32]);

        [1usize, 2, 3].iter().for_each(|a_cols| {
            [1usize, 2, 3].iter().for_each(|b_cols| {
                let min_cols: usize = min(*b_cols, *a_cols);
                let min_size: usize = min(b_size, a_size);

                // Allocats a and populates with random values.
                let mut a: VecZnx = module.new_vec_znx(*a_cols, a_size);
                (0..*a_cols).for_each(|i| {
                    module.fill_uniform(3, &mut a, i, a_size, &mut source);
                });

                // Allocats b and populates with random values.
                let mut b_have: VecZnx = module.new_vec_znx(*b_cols, b_size);
                (0..*b_cols).for_each(|i| {
                    module.fill_uniform(3, &mut b_have, i, b_size, &mut source);
                });

                let mut b_want: VecZnx = module.new_vec_znx(*b_cols, b_size);
                b_want.raw_mut().copy_from_slice(b_have.raw());

                // Applies the function to test.
                func_have(&mut b_have, &a);

                // Applies the reference function and expected behavior.
                // Applies with the minimum matching columns
                (0..min_cols).for_each(|i| {
                    // Adds with th eminimum matching size
                    (0..min_size).for_each(|j| func_want(b_want.at_poly_mut(i, j), a.at_poly(i, j)));
                    if NEGATE {
                        (min_size..b_size).for_each(|j| {
                            b_want.at_poly_mut(i, j).iter_mut().for_each(|x| *x = -*x);
                        });
                    }
                });

                assert_eq!(b_have.raw(), b_want.raw());
            });
        });
    }

    fn test_unary_op<B: Backend>(
        module: &Module<B>,
        func_have: impl Fn(&mut VecZnx, &VecZnx),
        func_want: impl Fn(&mut [i64], &[i64]),
    ) {
        let a_size: usize = 3;
        let b_size: usize = 5;
        let mut source = Source::new([0u8; 32]);

        [1usize, 2, 3].iter().for_each(|a_cols| {
            [1usize, 2, 3].iter().for_each(|b_cols| {
                let min_cols: usize = min(*b_cols, *a_cols);
                let min_size: usize = min(b_size, a_size);

                // Allocats a and populates with random values.
                let mut a: VecZnx = module.new_vec_znx(*a_cols, a_size);
                (0..a.cols()).for_each(|i| {
                    module.fill_uniform(3, &mut a, i, a_size, &mut source);
                });

                // Allocats b and populates with random values.
                let mut b_have: VecZnx = module.new_vec_znx(*b_cols, b_size);
                (0..b_have.cols()).for_each(|i| {
                    module.fill_uniform(3, &mut b_have, i, b_size, &mut source);
                });

                let mut b_want: VecZnx = module.new_vec_znx(*b_cols, b_size);

                // Applies the function to test.
                func_have(&mut b_have, &a);

                // Applies the reference function and expected behavior.
                // Applies on the minimum matching columns
                (0..min_cols).for_each(|i| {
                    // Applies on the minimum matching size
                    (0..min_size).for_each(|j| func_want(b_want.at_poly_mut(i, j), a.at_poly(i, j)));

                    // Zeroes the unmatching size
                    (min_size..b_size).for_each(|j| {
                        b_want.zero_at(i, j);
                    })
                });

                // Zeroes the unmatching columns
                (min_cols..*b_cols).for_each(|i| {
                    (0..b_size).for_each(|j| {
                        b_want.zero_at(i, j);
                    })
                });

                assert_eq!(b_have.raw(), b_want.raw());
            });
        });
    }

    fn test_unary_op_inplace<B: Backend>(module: &Module<B>, func_have: impl Fn(&mut VecZnx), func_want: impl Fn(&mut [i64])) {
        let a_size: usize = 3;
        let mut source = Source::new([0u8; 32]);
        [1usize, 2, 3].iter().for_each(|a_cols| {
            let mut a_have: VecZnx = module.new_vec_znx(*a_cols, a_size);
            (0..*a_cols).for_each(|i| {
                module.fill_uniform(3, &mut a_have, i, a_size, &mut source);
            });

            // Allocats a and populates with random values.
            let mut a_want: VecZnx = module.new_vec_znx(*a_cols, a_size);
            a_have.raw_mut().copy_from_slice(a_want.raw());

            // Applies the function to test.
            func_have(&mut a_have);

            // Applies the reference function and expected behavior.
            // Applies on the minimum matching columns
            (0..*a_cols).for_each(|i| {
                // Applies on the minimum matching size
                (0..a_size).for_each(|j| func_want(a_want.at_poly_mut(i, j)));
            });

            assert_eq!(a_have.raw(), a_want.raw());
        });
    }
}
