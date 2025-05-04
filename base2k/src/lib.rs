pub mod encoding;
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, improper_ctypes)]
// Other modules and exports
pub mod ffi;
pub mod mat_znx_dft;
pub mod mat_znx_dft_ops;
pub mod module;
pub mod sampling;
pub mod scalar_znx;
pub mod scalar_znx_dft;
pub mod scalar_znx_dft_ops;
pub mod stats;
pub mod vec_znx;
pub mod vec_znx_big;
pub mod vec_znx_big_ops;
pub mod vec_znx_dft;
pub mod vec_znx_dft_ops;
pub mod vec_znx_ops;
pub mod znx_base;

use std::{
    any::type_name,
    ops::{DerefMut, Sub},
};

pub use encoding::*;
pub use mat_znx_dft::*;
pub use mat_znx_dft_ops::*;
pub use module::*;
use rand_core::le;
use rand_distr::num_traits::sign;
pub use sampling::*;
pub use scalar_znx::*;
pub use scalar_znx_dft::*;
pub use scalar_znx_dft_ops::*;
pub use stats::*;
pub use vec_znx::*;
pub use vec_znx_big::*;
pub use vec_znx_big_ops::*;
pub use vec_znx_dft::*;
pub use vec_znx_dft_ops::*;
pub use vec_znx_ops::*;
pub use znx_base::*;

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
    let len: usize = data.len() / size_of::<V>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

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
        (size * size_of::<T>()) % align,
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
/// of [DEFAULTALIGN]/size_of::<T>() that is equal or greater to `size`.
pub fn alloc_aligned<T>(size: usize) -> Vec<T> {
    alloc_aligned_custom::<T>(
        size + (size % (DEFAULTALIGN / size_of::<T>())),
        DEFAULTALIGN,
    )
}

pub struct ScratchOwned(Vec<u8>);

impl ScratchOwned {
    pub fn new(byte_count: usize) -> Self {
        let data: Vec<u8> = alloc_aligned(byte_count);
        Self(data)
    }

    pub fn borrow(&mut self) -> &mut ScratchBorr {
        ScratchBorr::new(&mut self.0)
    }
}

pub struct ScratchBorr {
    data: [u8],
}

impl ScratchBorr {
    fn new(data: &mut [u8]) -> &mut Self {
        unsafe { &mut *(data as *mut [u8] as *mut Self) }
    }

    fn take_slice_aligned(data: &mut [u8], take_len: usize) -> (&mut [u8], &mut [u8]) {
        let ptr = data.as_mut_ptr();
        let self_len = data.len();

        let aligned_offset = ptr.align_offset(DEFAULTALIGN);
        let aligned_len = self_len.saturating_sub(aligned_offset);

        if let Some(rem_len) = aligned_len.checked_sub(take_len) {
            unsafe {
                let rem_ptr = ptr.add(aligned_offset).add(take_len);
                let rem_slice = &mut *std::ptr::slice_from_raw_parts_mut(rem_ptr, rem_len);

                let take_slice = &mut *std::ptr::slice_from_raw_parts_mut(ptr.add(aligned_offset), take_len);

                return (take_slice, rem_slice);
            }
        } else {
            panic!(
                "Attempted to take {} from scratch with {} aligned bytes left",
                take_len,
                take_len,
                // type_name::<T>(),
                // aligned_len
            );
        }
    }

    fn tmp_scalar_slice<T>(&mut self, len: usize) -> (&mut [T], &mut Self) {
        let (take_slice, rem_slice) = Self::take_slice_aligned(&mut self.data, len * std::mem::size_of::<T>());

        unsafe {
            (
                &mut *(std::ptr::slice_from_raw_parts_mut(take_slice.as_mut_ptr() as *mut T, len)),
                Self::new(rem_slice),
            )
        }
    }

    fn tmp_vec_znx_dft<B: Backend>(
        &mut self,
        module: &Module<B>,
        cols: usize,
        size: usize,
    ) -> (VecZnxDft<&mut [u8], B>, &mut Self) {
        let (take_slice, rem_slice) = Self::take_slice_aligned(&mut self.data, bytes_of_vec_znx_dft(module, cols, size));

        (
            VecZnxDft::from_data(take_slice, module.n(), cols, size),
            Self::new(rem_slice),
        )
    }

    fn tmp_vec_znx_big<D: for<'a> From<&'a mut [u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        cols: usize,
        size: usize,
    ) -> (VecZnxBig<D, B>, &mut Self) {
        let (take_slice, rem_slice) = Self::take_slice_aligned(&mut self.data, bytes_of_vec_znx_big(module, cols, size));

        (
            VecZnxBig::from_data(D::from(take_slice), module.n(), cols, size),
            Self::new(rem_slice),
        )
    }
}

// pub struct ScratchBorrowed<'a> {
//     data: &'a mut [u8],
// }

// impl<'a> ScratchBorrowed<'a> {
//     fn take_slice<T>(&mut self, take_len: usize) -> (&mut [T], ScratchBorrowed<'_>) {
//         let ptr = self.data.as_mut_ptr();
//         let self_len = self.data.len();

//         //TODO(Jay): print the offset sometimes, just to check
//         let aligned_offset = ptr.align_offset(DEFAULTALIGN);
//         let aligned_len = self_len.saturating_sub(aligned_offset);

//         let take_len_bytes = take_len * std::mem::size_of::<T>();

//         if let Some(rem_len) = aligned_len.checked_sub(take_len_bytes) {
//             unsafe {
//                 let rem_ptr = ptr.add(aligned_offset).add(take_len_bytes);
//                 let rem_slice = &mut *std::ptr::slice_from_raw_parts_mut(rem_ptr, rem_len);

//                 let take_slice = &mut *std::ptr::slice_from_raw_parts_mut(ptr.add(aligned_offset) as *mut T, take_len_bytes);

//                 return (take_slice, ScratchBorrowed { data: rem_slice });
//             }
//         } else {
//             panic!(
//                 "Attempted to take {} (={} elements of {}) from scratch with {} aligned bytes left",
//                 take_len_bytes,
//                 take_len,
//                 type_name::<T>(),
//                 aligned_len
//             );
//         }
//     }

//     fn reborrow(&mut self) -> ScratchBorrowed<'a> {
//         //(Jay)TODO: `data: &mut *self.data` does not work because liftime of &mut self is different from 'a.
//         // But it feels that there should be a simpler impl. than the one below
//         Self {
//             data: unsafe { &mut *std::ptr::slice_from_raw_parts_mut(self.data.as_mut_ptr(), self.data.len()) },
//         }
//     }

//     fn tmp_vec_znx_dft<B: Backend>(&mut self, module: &Module<B>, cols: usize, size: usize) -> (VecZnxDft<&mut [u8], B>, Self) {
//         let (data, re_scratch) = self.take_slice::<u8>(vec_znx_dft::bytes_of_vec_znx_dft(module, cols, size));
//         (
//             VecZnxDft::from_data(data, module.n(), cols, size),
//             re_scratch,
//         )
//     }

//     pub(crate) fn len(&self) -> usize {
//         self.data.len()
//     }
// }

// pub trait Scratch<D> {
//     fn tmp_vec_znx_dft<B: Backend>(&mut self, module: &Module<B>, cols: usize, size: usize) -> (D, &mut Self);
// }

// impl<'a> Scratch<&'a mut [u8]> for ScratchBorr {
//     fn tmp_vec_znx_dft<B: Backend>(&mut self, module: &Module<B>, cols: usize, size: usize) -> (&'a mut [u8], &mut Self) {
//         let (data, rem_scratch) = self.tmp_scalar_slice(vec_znx_dft::bytes_of_vec_znx_dft(module, cols, size));
//         (
//             data
//             rem_scratch,
//         )
//     }

//     // fn tmp_vec_znx_big<B: Backend>(&mut self, module: &Module<B>, cols: usize, size: usize) -> (VecZnxBig<&mut [u8], B>, Self) {
//     //     // let (data, re_scratch) = self.take_slice(vec_znx_big::bytes_of_vec_znx_big(module, cols, size));
//     //     // (
//     //     //     VecZnxBig::from_data(data, module.n(), cols, size),
//     //     //     re_scratch,
//     //     // )
//     // }

//     // fn scalar_slice<T>(&mut self, len: usize) -> (&mut [T], Self) {
//     //     self.take_slice::<T>(len)
//     // }

//     // fn reborrow(&mut self) -> Self {
//     //     self.reborrow()
//     // }
// }
