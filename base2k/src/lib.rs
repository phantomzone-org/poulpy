pub mod commons;
pub mod encoding;
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, improper_ctypes)]
// Other modules and exports
pub mod ffi;
mod internals;
pub mod mat_znx_dft;
pub mod module;
pub mod sampling;
pub mod scalar_znx_dft;
pub mod stats;
pub mod vec_znx;
pub mod vec_znx_big;
pub mod vec_znx_big_ops;
pub mod vec_znx_dft;
pub mod vec_znx_ops;

pub use commons::*;
pub use encoding::*;
pub use mat_znx_dft::*;
pub use module::*;
pub use sampling::*;
pub use scalar_znx_dft::*;
#[allow(unused_imports)]
pub use stats::*;
pub use vec_znx::*;
pub use vec_znx_big::*;
pub use vec_znx_big_ops::*;
pub use vec_znx_dft::*;
pub use vec_znx_ops::*;

pub const GALOISGENERATOR: u64 = 5;
pub const DEFAULTALIGN: usize = 64;

fn is_aligned_custom<T>(ptr: *const T, align: usize) -> bool {
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

/// Allocates a block of bytes with a custom alignement.
/// Alignement must be a power of two and size a multiple of the alignement.
/// Allocated memory is initialized to zero.
fn alloc_aligned_custom_u8(size: usize, align: usize) -> Vec<u8> {
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
        let layout: std::alloc::Layout = std::alloc::Layout::from_size_align(size, align).expect("Invalid alignment");
        let ptr: *mut u8 = std::alloc::alloc(layout);
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
        std::ptr::write_bytes(ptr, 0, size);
        Vec::from_raw_parts(ptr, size, size)
    }
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

// Allocates an aligned of size equal to the smallest power of two equal or greater to `size` that is
// at least as bit as DEFAULTALIGN / std::mem::size_of::<T>().
pub fn alloc_aligned<T>(size: usize) -> Vec<T> {
    alloc_aligned_custom::<T>(
        std::cmp::max(
            size.next_power_of_two(),
            DEFAULTALIGN / std::mem::size_of::<T>(),
        ),
        DEFAULTALIGN,
    )
}
