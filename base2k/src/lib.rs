pub mod encoding;
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, improper_ctypes)]
// Other modules and exports
pub mod ffi;
pub mod infos;
pub mod module;
pub mod sampling;
pub mod stats;
pub mod svp;
pub mod vec_znx;
pub mod vec_znx_big;
pub mod vec_znx_dft;
pub mod vmp;

pub use encoding::*;
pub use infos::*;
pub use module::*;
pub use sampling::*;
#[allow(unused_imports)]
pub use stats::*;
pub use svp::*;
pub use vec_znx::*;
pub use vec_znx_big::*;
pub use vec_znx_dft::*;
pub use vmp::*;

pub const GALOISGENERATOR: u64 = 5;
pub const DEFAULTALIGN: usize = 64;

pub fn is_aligned_custom<T>(ptr: *const T, align: usize) -> bool {
    (ptr as usize) % align == 0
}

pub fn is_aligned<T>(ptr: *const T) -> bool {
    is_aligned_custom(ptr, DEFAULTALIGN)
}

pub fn assert_alignement<T>(ptr: *const T) {
    assert!(
        is_aligned(ptr),
        "invalid alignement: ensure passed bytes have been allocated with [alloc_aligned_u8] or [alloc_aligned]"
    )
}

pub fn cast<T, V>(data: &[T]) -> &[V] {
    let ptr: *const V = data.as_ptr() as *const V;
    let len: usize = data.len() / std::mem::size_of::<V>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

pub fn cast_mut<T, V>(data: &[T]) -> &mut [V] {
    let ptr: *mut V = data.as_ptr() as *mut V;
    let len: usize = data.len() / std::mem::size_of::<V>();
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

use std::alloc::{Layout, alloc};
use std::ptr;

/// Allocates a block of bytes with a custom alignement.
/// Alignement must be a power of two and size a multiple of the alignement.
/// Allocated memory is initialized to zero.
pub fn alloc_aligned_custom_u8(size: usize, align: usize) -> Vec<u8> {
    assert!(
        align.is_power_of_two(),
        "Alignment must be a power of two but is {}",
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
        assert!(
            is_aligned_custom(ptr, align),
            "Memory allocation at {:p} is not aligned to {} bytes",
            ptr,
            align
        );
        // Init allocated memory to zero
        ptr::write_bytes(ptr, 0, size);
        Vec::from_raw_parts(ptr, size, size)
    }
}

/// Allocates a block of bytes aligned with [DEFAULTALIGN].
/// Size must be amultiple of [DEFAULTALIGN].
/// /// Allocated memory is initialized to zero.
pub fn alloc_aligned_u8(size: usize) -> Vec<u8> {
    alloc_aligned_custom_u8(size, DEFAULTALIGN)
}

/// Allocates a block of T aligned with [DEFAULTALIGN].
/// Size of T * size msut be a multiple of [DEFAULTALIGN].
pub fn alloc_aligned_custom<T>(size: usize, align: usize) -> Vec<T> {
    assert_eq!(
        (size * std::mem::size_of::<T>()) % align,
        0,
        "size={} must be a multiple of align={}",
        size,
        align
    );
    let mut vec_u8: Vec<u8> = alloc_aligned_custom_u8(std::mem::size_of::<T>() * size, align);
    let ptr: *mut T = vec_u8.as_mut_ptr() as *mut T;
    let len: usize = vec_u8.len() / std::mem::size_of::<T>();
    let cap: usize = vec_u8.capacity() / std::mem::size_of::<T>();
    std::mem::forget(vec_u8);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

pub fn alloc_aligned<T>(size: usize) -> Vec<T> {
    alloc_aligned_custom::<T>(size, DEFAULTALIGN)
}
