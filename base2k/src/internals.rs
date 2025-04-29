use std::cmp::{max, min};

use crate::{Backend, IntegerType, Module, ZnxBasics, ZnxLayout, ffi::module::MODULE};

pub(crate) fn znx_post_process_ternary_op<C, A, B, const NEGATE: bool>(c: &mut C, a: &A, b: &B)
where
    C: ZnxBasics + ZnxLayout,
    A: ZnxBasics + ZnxLayout<Scalar = C::Scalar>,
    B: ZnxBasics + ZnxLayout<Scalar = C::Scalar>,
    C::Scalar: IntegerType,
{
    #[cfg(debug_assertions)]
    {
        assert_ne!(a.as_ptr(), b.as_ptr());
        assert_ne!(b.as_ptr(), c.as_ptr());
        assert_ne!(a.as_ptr(), c.as_ptr());
    }

    let a_cols: usize = a.cols();
    let b_cols: usize = b.cols();
    let c_cols: usize = c.cols();

    let min_ab_cols: usize = min(a_cols, b_cols);
    let max_ab_cols: usize = max(a_cols, b_cols);

    // Copies shared shared cols between (c, max(a, b))
    if a_cols != b_cols {
        if a_cols > b_cols {
            let min_size = min(c.size(), a.size());
            (min_ab_cols..min(max_ab_cols, c_cols)).for_each(|i| {
                (0..min_size).for_each(|j| {
                    c.at_poly_mut(i, j).copy_from_slice(a.at_poly(i, j));
                    if NEGATE {
                        c.at_poly_mut(i, j).iter_mut().for_each(|x| *x = -*x);
                    }
                });
                (min_size..c.size()).for_each(|j| {
                    c.zero_at(i, j);
                });
            });
        } else {
            let min_size = min(c.size(), b.size());
            (min_ab_cols..min(max_ab_cols, c_cols)).for_each(|i| {
                (0..min_size).for_each(|j| {
                    c.at_poly_mut(i, j).copy_from_slice(b.at_poly(i, j));
                    if NEGATE {
                        c.at_poly_mut(i, j).iter_mut().for_each(|x| *x = -*x);
                    }
                });
                (min_size..c.size()).for_each(|j| {
                    c.zero_at(i, j);
                });
            });
        }
    }

    // Zeroes the cols of c > max(a, b).
    if c_cols > max_ab_cols {
        (max_ab_cols..c_cols).for_each(|i| {
            (0..c.size()).for_each(|j| {
                c.zero_at(i, j);
            })
        });
    }
}

#[inline(always)]
pub fn apply_binary_op<BE, C, A, B, const NEGATE: bool>(
    module: &Module<BE>,
    c: &mut C,
    a: &A,
    b: &B,
    op: impl Fn(&mut [C::Scalar], &[A::Scalar], &[B::Scalar]),
) where
    BE: Backend,
    C: ZnxBasics + ZnxLayout,
    A: ZnxBasics + ZnxLayout<Scalar = C::Scalar>,
    B: ZnxBasics + ZnxLayout<Scalar = C::Scalar>,
    C::Scalar: IntegerType,
{
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
        znx_post_process_ternary_op::<C, A, B, NEGATE>(c, a, b);
    }
}

#[inline(always)]
pub fn apply_unary_op<B: Backend, T: ZnxBasics + ZnxLayout>(
    module: &Module<B>,
    b: &mut T,
    a: &T,
    op: impl Fn(&mut [T::Scalar], &[T::Scalar]),
) where
    <T as ZnxLayout>::Scalar: IntegerType,
{
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

pub fn ffi_ternary_op_factory<T>(
    module_ptr: *const MODULE,
    c_size: usize,
    c_sl: usize,
    a_size: usize,
    a_sl: usize,
    b_size: usize,
    b_sl: usize,
    op_fn: unsafe extern "C" fn(*const MODULE, *mut T, u64, u64, *const T, u64, u64, *const T, u64, u64),
) -> impl Fn(&mut [T], &[T], &[T]) {
    move |cv: &mut [T], av: &[T], bv: &[T]| unsafe {
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

pub fn ffi_binary_op_factory_type_0<T>(
    module_ptr: *const MODULE,
    b_size: usize,
    b_sl: usize,
    a_size: usize,
    a_sl: usize,
    op_fn: unsafe extern "C" fn(*const MODULE, *mut T, u64, u64, *const T, u64, u64),
) -> impl Fn(&mut [T], &[T]) {
    move |bv: &mut [T], av: &[T]| unsafe {
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

pub fn ffi_binary_op_factory_type_1<T>(
    module_ptr: *const MODULE,
    k: i64,
    b_size: usize,
    b_sl: usize,
    a_size: usize,
    a_sl: usize,
    op_fn: unsafe extern "C" fn(*const MODULE, i64, *mut T, u64, u64, *const T, u64, u64),
) -> impl Fn(&mut [T], &[T]) {
    move |bv: &mut [T], av: &[T]| unsafe {
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
