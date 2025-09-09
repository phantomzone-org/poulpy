#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, improper_ctypes)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![feature(trait_alias)]

pub mod api;
pub mod delegates;
pub mod layouts;
pub mod oep;
pub mod reference;
pub mod source;
pub mod tests;

pub mod doc {
    #[doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/backend_safety_contract.md"))]
    pub mod backend_safety {
        pub const _PLACEHOLDER: () = ();
    }
}

pub const GALOISGENERATOR: u64 = 5;
pub const DEFAULTALIGN: usize = 64;

fn is_aligned_custom<T>(ptr: *const T, align: usize) -> bool {
    (ptr as usize).is_multiple_of(align)
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
    let len: usize = data.len() / size_of::<V>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

#[allow(clippy::mut_from_ref)]
pub fn cast_mut<T, V>(data: &[T]) -> &mut [V] {
    let ptr: *mut V = data.as_ptr() as *mut V;
    let len: usize = data.len() / size_of::<V>();
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
        (size * size_of::<u8>()) % align,
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
        (size * size_of::<T>()) % (align / size_of::<T>()),
        0,
        "size={} must be a multiple of align={}",
        size,
        align
    );
    let mut vec_u8: Vec<u8> = alloc_aligned_custom_u8(size_of::<T>() * size, align);
    let ptr: *mut T = vec_u8.as_mut_ptr() as *mut T;
    let len: usize = vec_u8.len() / size_of::<T>();
    let cap: usize = vec_u8.capacity() / size_of::<T>();
    std::mem::forget(vec_u8);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

/// Allocates an aligned vector of size equal to the smallest multiple
/// of [DEFAULTALIGN]/`size_of::<T>`() that is equal or greater to `size`.
pub fn alloc_aligned<T>(size: usize) -> Vec<T> {
    alloc_aligned_custom::<T>(
        size + (DEFAULTALIGN - (size % (DEFAULTALIGN / size_of::<T>()))) % DEFAULTALIGN,
        DEFAULTALIGN,
    )
}
