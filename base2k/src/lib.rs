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

pub mod vec_znx;
#[allow(unused_imports)]
pub use vec_znx::*;

pub mod vec_znx_big;
#[allow(unused_imports)]
pub use vec_znx_big::*;

pub mod vec_znx_dft;
#[allow(unused_imports)]
pub use vec_znx_dft::*;

pub mod svp;
#[allow(unused_imports)]
pub use svp::*;

pub mod vmp;
#[allow(unused_imports)]
pub use vmp::*;

pub mod sampling;
#[allow(unused_imports)]
pub use sampling::*;

pub mod encoding;
#[allow(unused_imports)]
pub use encoding::*;

pub mod infos;
#[allow(unused_imports)]
pub use infos::*;

pub mod free;
#[allow(unused_imports)]
pub use free::*;

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

use std::alloc::{alloc, Layout};

pub fn alloc_aligned_u8(size: usize, align: usize) -> Vec<u8> {
    assert_eq!(
        align & (align - 1),
        0,
        "align={} must be a power of two",
        align
    );
    assert_eq!(
        (size * std::mem::size_of::<u8>()) % align,
        0,
        "size={} must be a multiple of align={}",
        size,
        align
    );
    unsafe {
        let layout: Layout = Layout::from_size_align(size, align).expect("Invalid alignment");
        let ptr: *mut u8 = alloc(layout);
        if ptr.is_null() {
            panic!("Memory allocation failed");
        }
        Vec::from_raw_parts(ptr, size, size)
    }
}

pub fn alloc_aligned<T>(size: usize, align: usize) -> Vec<T> {
    assert_eq!(
        (size * std::mem::size_of::<T>()) % align,
        0,
        "size={} must be a multiple of align={}",
        size,
        align
    );
    let mut vec_u8: Vec<u8> = alloc_aligned_u8(std::mem::size_of::<T>() * size, align);
    let ptr: *mut T = vec_u8.as_mut_ptr() as *mut T;
    let len: usize = vec_u8.len() / std::mem::size_of::<T>();
    let cap: usize = vec_u8.capacity() / std::mem::size_of::<T>();
    std::mem::forget(vec_u8);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

fn alias_mut_slice_to_vec<T>(slice: &mut [T]) -> Vec<T> {
    let ptr = slice.as_mut_ptr();
    let len = slice.len();
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}
