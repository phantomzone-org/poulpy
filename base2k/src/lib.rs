#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    improper_ctypes
)]
pub mod ffi;

pub mod module;
#[allow(unused_imports)]
pub use module::*;

pub mod scalar;
#[allow(unused_imports)]
pub use scalar::*;

pub mod vec_znx;
#[allow(unused_imports)]
pub use vec_znx::*;

pub mod vec_znx_arithmetic;
#[allow(unused_imports)]
pub use vec_znx_arithmetic::*;

pub mod vec_znx_big_arithmetic;
#[allow(unused_imports)]
pub use vec_znx_big_arithmetic::*;

pub mod vec_znx_dft;
#[allow(unused_imports)]
pub use vec_znx_dft::*;

pub mod svp;
#[allow(unused_imports)]
pub use svp::*;

pub mod vmp;
#[allow(unused_imports)]
pub use vmp::*;

pub const GALOISGENERATOR: u64 = 5;

#[allow(dead_code)]
pub fn cast_mut_u64_to_mut_u8_slice(data: &mut [u64]) -> &mut [u8] {
    let ptr: *mut u8 = data.as_mut_ptr() as *mut u8;
    let len: usize = data.len() * std::mem::size_of::<u64>();
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

pub fn cast_mut_u8_to_mut_i64_slice(data: &mut [u8]) -> &mut [i64] {
    let ptr: *mut i64 = data.as_mut_ptr() as *mut i64;
    let len: usize = data.len() / std::mem::size_of::<i64>();
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

pub fn cast_mut_u8_to_mut_f64_slice(data: &mut [u8]) -> &mut [f64] {
    let ptr: *mut f64 = data.as_mut_ptr() as *mut f64;
    let len: usize = data.len() / std::mem::size_of::<f64>();
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

pub fn cast_u8_to_f64_slice(data: &mut [u8]) -> &[f64] {
    let ptr: *const f64 = data.as_mut_ptr() as *const f64;
    let len: usize = data.len() / std::mem::size_of::<f64>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// This trait should be implemented by structs that point to
/// memory allocated through C.
pub trait Free {
    // Frees the memory and self destructs.
    fn free(self);
}
