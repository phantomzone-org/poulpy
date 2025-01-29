#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    improper_ctypes
)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

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

pub mod scalar_vector_product;
#[allow(unused_imports)]
pub use scalar_vector_product::*;

pub mod vector_matrix_product;
#[allow(unused_imports)]
pub use vector_matrix_product::*;

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
