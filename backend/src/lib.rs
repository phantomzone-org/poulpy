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

pub use encoding::*;
pub use mat_znx_dft::*;
pub use mat_znx_dft_ops::*;
pub use module::*;
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
/// of [DEFAULTALIGN]/size_of::<T>() that is equal or greater to `size`.
pub fn alloc_aligned<T>(size: usize) -> Vec<T> {
    alloc_aligned_custom::<T>(
        size + (DEFAULTALIGN - (size % (DEFAULTALIGN / size_of::<T>()))),
        DEFAULTALIGN,
    )
}

// Scratch implementation below

pub struct ScratchOwned(Vec<u8>);

impl ScratchOwned {
    pub fn new(byte_count: usize) -> Self {
        let data: Vec<u8> = alloc_aligned(byte_count);
        Self(data)
    }

    pub fn borrow(&mut self) -> &mut Scratch {
        Scratch::new(&mut self.0)
    }
}

pub struct Scratch {
    data: [u8],
}

impl Scratch {
    fn new(data: &mut [u8]) -> &mut Self {
        unsafe { &mut *(data as *mut [u8] as *mut Self) }
    }

    pub fn zero(&mut self) {
        self.data.fill(0);
    }

    pub fn available(&self) -> usize {
        let ptr: *const u8 = self.data.as_ptr();
        let self_len: usize = self.data.len();
        let aligned_offset: usize = ptr.align_offset(DEFAULTALIGN);
        self_len.saturating_sub(aligned_offset)
    }

    fn take_slice_aligned(data: &mut [u8], take_len: usize) -> (&mut [u8], &mut [u8]) {
        let ptr: *mut u8 = data.as_mut_ptr();
        let self_len: usize = data.len();

        let aligned_offset: usize = ptr.align_offset(DEFAULTALIGN);
        let aligned_len: usize = self_len.saturating_sub(aligned_offset);

        if let Some(rem_len) = aligned_len.checked_sub(take_len) {
            unsafe {
                let rem_ptr: *mut u8 = ptr.add(aligned_offset).add(take_len);
                let rem_slice: &mut [u8] = &mut *std::ptr::slice_from_raw_parts_mut(rem_ptr, rem_len);

                let take_slice: &mut [u8] = &mut *std::ptr::slice_from_raw_parts_mut(ptr.add(aligned_offset), take_len);

                return (take_slice, rem_slice);
            }
        } else {
            panic!(
                "Attempted to take {} from scratch with {} aligned bytes left",
                take_len,
                aligned_len,
                // type_name::<T>(),
                // aligned_len
            );
        }
    }

    pub fn tmp_slice<T>(&mut self, len: usize) -> (&mut [T], &mut Self) {
        let (take_slice, rem_slice) = Self::take_slice_aligned(&mut self.data, len * std::mem::size_of::<T>());

        unsafe {
            (
                &mut *(std::ptr::slice_from_raw_parts_mut(take_slice.as_mut_ptr() as *mut T, len)),
                Self::new(rem_slice),
            )
        }
    }

    pub fn tmp_scalar_znx<B: Backend>(&mut self, module: &Module<B>, cols: usize) -> (ScalarZnx<&mut [u8]>, &mut Self) {
        let (take_slice, rem_slice) = Self::take_slice_aligned(&mut self.data, bytes_of_scalar_znx(module, cols));

        (
            ScalarZnx::from_data(take_slice, module.n(), cols),
            Self::new(rem_slice),
        )
    }

    pub fn tmp_scalar_znx_dft<B: Backend>(&mut self, module: &Module<B>, cols: usize) -> (ScalarZnxDft<&mut [u8], B>, &mut Self) {
        let (take_slice, rem_slice) = Self::take_slice_aligned(&mut self.data, bytes_of_scalar_znx_dft(module, cols));

        (
            ScalarZnxDft::from_data(take_slice, module.n(), cols),
            Self::new(rem_slice),
        )
    }

    pub fn tmp_vec_znx_dft<B: Backend>(
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

    pub fn tmp_slice_vec_znx_dft<B: Backend>(
        &mut self,
        slice_size: usize,
        module: &Module<B>,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnxDft<&mut [u8], B>>, &mut Self) {
        let mut scratch: &mut Scratch = self;
        let mut slice: Vec<VecZnxDft<&mut [u8], B>> = Vec::with_capacity(slice_size);
        for _ in 0..slice_size {
            let (znx, new_scratch) = scratch.tmp_vec_znx_dft(module, cols, size);
            scratch = new_scratch;
            slice.push(znx);
        }
        (slice, scratch)
    }

    pub fn tmp_vec_znx_big<B: Backend>(
        &mut self,
        module: &Module<B>,
        cols: usize,
        size: usize,
    ) -> (VecZnxBig<&mut [u8], B>, &mut Self) {
        let (take_slice, rem_slice) = Self::take_slice_aligned(&mut self.data, bytes_of_vec_znx_big(module, cols, size));

        (
            VecZnxBig::from_data(take_slice, module.n(), cols, size),
            Self::new(rem_slice),
        )
    }

    pub fn tmp_vec_znx<B: Backend>(&mut self, module: &Module<B>, cols: usize, size: usize) -> (VecZnx<&mut [u8]>, &mut Self) {
        let (take_slice, rem_slice) = Self::take_slice_aligned(&mut self.data, module.bytes_of_vec_znx(cols, size));
        (
            VecZnx::from_data(take_slice, module.n(), cols, size),
            Self::new(rem_slice),
        )
    }

    pub fn tmp_slice_vec_znx<B: Backend>(
        &mut self,
        slice_size: usize,
        module: &Module<B>,
        cols: usize,
        size: usize,
    ) -> (Vec<VecZnx<&mut [u8]>>, &mut Self) {
        let mut scratch: &mut Scratch = self;
        let mut slice: Vec<VecZnx<&mut [u8]>> = Vec::with_capacity(slice_size);
        for _ in 0..slice_size {
            let (znx, new_scratch) = scratch.tmp_vec_znx(module, cols, size);
            scratch = new_scratch;
            slice.push(znx);
        }
        (slice, scratch)
    }

    pub fn tmp_mat_znx_dft<B: Backend>(
        &mut self,
        module: &Module<B>,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> (MatZnxDft<&mut [u8], B>, &mut Self) {
        let (take_slice, rem_slice) = Self::take_slice_aligned(
            &mut self.data,
            module.bytes_of_mat_znx_dft(rows, cols_in, cols_out, size),
        );
        (
            MatZnxDft::from_data(take_slice, module.n(), rows, cols_in, cols_out, size),
            Self::new(rem_slice),
        )
    }
}
