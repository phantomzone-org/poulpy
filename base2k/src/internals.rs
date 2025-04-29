use std::cmp::{max, min};

use crate::{Backend, IntegerType, Module, ZnxBasics, ZnxLayout, ffi::module::MODULE};

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
