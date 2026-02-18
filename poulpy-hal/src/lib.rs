#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, improper_ctypes)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![feature(trait_alias)]

pub mod api;
pub mod bench_suite;
pub mod delegates;
pub mod layouts;
pub mod oep;
pub mod reference;
pub mod source;
pub mod test_suite;

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

pub fn assert_alignment<T>(ptr: *const T) {
    assert!(
        is_aligned(ptr),
        "invalid alignment: ensure passed bytes have been allocated with [alloc_aligned_u8] or [alloc_aligned]"
    )
}

/// Deprecated spelling variant. Use [assert_alignment] instead.
#[inline]
pub fn assert_alignement<T>(ptr: *const T) {
    assert_alignment(ptr)
}

/// Reinterprets a `&[T]` as a `&[V]`.
///
/// # Safety (via assertions)
/// - `V` must not be zero-sized.
/// - The pointer must be aligned for `V`.
/// - The total byte length must be a multiple of `size_of::<V>()`.
pub fn cast<T, V>(data: &[T]) -> &[V] {
    assert!(size_of::<V>() > 0, "cast: target type V must not be zero-sized");
    let byte_len: usize = data.len() * size_of::<T>();
    assert!(
        byte_len % size_of::<V>() == 0,
        "cast: byte length {} is not a multiple of target size {}",
        byte_len,
        size_of::<V>()
    );
    let ptr: *const V = data.as_ptr() as *const V;
    assert!(
        ptr.align_offset(align_of::<V>()) == 0,
        "cast: pointer {:p} is not aligned to {} bytes",
        ptr,
        align_of::<V>()
    );
    let len: usize = byte_len / size_of::<V>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Reinterprets a `&mut [T]` as a `&mut [V]`.
///
/// # Safety (via assertions)
/// - `V` must not be zero-sized.
/// - The pointer must be aligned for `V`.
/// - The total byte length must be a multiple of `size_of::<V>()`.
pub fn cast_mut<T, V>(data: &mut [T]) -> &mut [V] {
    assert!(size_of::<V>() > 0, "cast_mut: target type V must not be zero-sized");
    let byte_len: usize = data.len() * size_of::<T>();
    assert!(
        byte_len % size_of::<V>() == 0,
        "cast_mut: byte length {} is not a multiple of target size {}",
        byte_len,
        size_of::<V>()
    );
    let ptr: *mut V = data.as_mut_ptr() as *mut V;
    assert!(
        ptr.align_offset(align_of::<V>()) == 0,
        "cast_mut: pointer {:p} is not aligned to {} bytes",
        ptr,
        align_of::<V>()
    );
    let len: usize = byte_len / size_of::<V>();
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

/// Allocates a block of bytes with a custom alignment.
/// Alignment must be a power of two and size a multiple of the alignment.
/// Allocated memory is initialized to zero.
///
/// # Known issue (CRITICAL-2)
/// The returned `Vec<u8>` was allocated with custom alignment via `std::alloc::alloc`,
/// but `Vec::drop` will call `std::alloc::dealloc` with `align_of::<u8>() = 1`.
/// This is technically UB per the `GlobalAlloc` contract (mismatched layout).
/// In practice it works on all major allocators (glibc, jemalloc, mimalloc) because
/// they ignore the alignment parameter during deallocation. A proper fix requires
/// replacing `Vec<u8>` with a custom `AlignedBuf` type that tracks the layout.
fn alloc_aligned_custom_u8(size: usize, align: usize) -> Vec<u8> {
    assert!(align.is_power_of_two(), "Alignment must be a power of two but is {align}");
    assert_eq!(
        (size * size_of::<u8>()) % align,
        0,
        "size={size} must be a multiple of align={align}"
    );
    unsafe {
        let layout: std::alloc::Layout = std::alloc::Layout::from_size_align(size, align).expect("Invalid alignment");
        let ptr: *mut u8 = std::alloc::alloc(layout);
        if ptr.is_null() {
            panic!("Memory allocation failed");
        }
        assert!(
            is_aligned_custom(ptr, align),
            "Memory allocation at {ptr:p} is not aligned to {align} bytes"
        );
        // Init allocated memory to zero
        std::ptr::write_bytes(ptr, 0, size);
        Vec::from_raw_parts(ptr, size, size)
    }
}

/// Allocates a block of T aligned with [DEFAULTALIGN].
/// Size of T * size must be a multiple of [DEFAULTALIGN].
///
/// # Panics
/// Panics if `T` is zero-sized.
pub fn alloc_aligned_custom<T>(size: usize, align: usize) -> Vec<T> {
    assert!(size_of::<T>() > 0, "alloc_aligned_custom: zero-sized types are not supported");
    assert!(
        align.is_power_of_two(),
        "Alignment must be a power of two but is {align}"
    );

    assert_eq!(
        (size * size_of::<T>()) % align,
        0,
        "size*size_of::<T>()={} must be a multiple of align={align}",
        size * size_of::<T>(),
    );

    let mut vec_u8: Vec<u8> = alloc_aligned_custom_u8(size_of::<T>() * size, align);
    let ptr: *mut T = vec_u8.as_mut_ptr() as *mut T;
    let len: usize = vec_u8.len() / size_of::<T>();
    let cap: usize = vec_u8.capacity() / size_of::<T>();
    std::mem::forget(vec_u8);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

/// Allocates an aligned vector of the given size.
/// Padds until it is size in [u8] a multiple of [DEFAULTALIGN].
pub fn alloc_aligned<T>(size: usize) -> Vec<T> {
    alloc_aligned_custom::<T>(
        (size * size_of::<T>()).next_multiple_of(DEFAULTALIGN) / size_of::<T>(),
        DEFAULTALIGN,
    )
}
