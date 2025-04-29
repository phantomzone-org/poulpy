use crate::ffi::vec_znx_big;
use crate::{Backend, FFT64, Module, ZnxBase, ZnxBasics, ZnxInfos, ZnxLayout, alloc_aligned, assert_alignement};
use std::marker::PhantomData;

pub struct VecZnxBig<B: Backend> {
    pub data: Vec<u8>,
    pub ptr: *mut u8,
    pub n: usize,
    pub cols: usize,
    pub size: usize,
    pub _marker: PhantomData<B>,
}

impl ZnxBasics for VecZnxBig<FFT64> {}

impl<B: Backend> ZnxBase<B> for VecZnxBig<B> {
    type Scalar = u8;

    fn new(module: &Module<B>, cols: usize, size: usize) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(cols > 0);
            assert!(size > 0);
        }
        let mut data: Vec<Self::Scalar> = alloc_aligned(Self::bytes_of(module, cols, size));
        let ptr: *mut Self::Scalar = data.as_mut_ptr();
        Self {
            data: data,
            ptr: ptr,
            n: module.n(),
            cols: cols,
            size: size,
            _marker: PhantomData,
        }
    }

    fn bytes_of(module: &Module<B>, cols: usize, size: usize) -> usize {
        unsafe { vec_znx_big::bytes_of_vec_znx_big(module.ptr, size as u64) as usize * cols }
    }

    /// Returns a new [VecZnxBig] with the provided data as backing array.
    /// User must ensure that data is properly alligned and that
    /// the size of data is at least equal to [Module::bytes_of_vec_znx_big].
    fn from_bytes(module: &Module<B>, cols: usize, size: usize, bytes: &mut [Self::Scalar]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(cols > 0);
            assert!(size > 0);
            assert_eq!(bytes.len(), Self::bytes_of(module, cols, size));
            assert_alignement(bytes.as_ptr())
        };
        unsafe {
            Self {
                data: Vec::from_raw_parts(bytes.as_mut_ptr(), bytes.len(), bytes.len()),
                ptr: bytes.as_mut_ptr(),
                n: module.n(),
                cols: cols,
                size: size,
                _marker: PhantomData,
            }
        }
    }

    fn from_bytes_borrow(module: &Module<B>, cols: usize, size: usize, bytes: &mut [Self::Scalar]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(cols > 0);
            assert!(size > 0);
            assert_eq!(bytes.len(), Self::bytes_of(module, cols, size));
            assert_alignement(bytes.as_ptr());
        }
        Self {
            data: Vec::new(),
            ptr: bytes.as_mut_ptr(),
            n: module.n(),
            cols: cols,
            size: size,
            _marker: PhantomData,
        }
    }
}

impl<B: Backend> ZnxInfos for VecZnxBig<B> {
    fn n(&self) -> usize {
        self.n
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn rows(&self) -> usize {
        1
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl ZnxLayout for VecZnxBig<FFT64> {
    type Scalar = i64;

    fn as_ptr(&self) -> *const Self::Scalar {
        self.ptr as *const Self::Scalar
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Scalar {
        self.ptr as *mut Self::Scalar
    }
}

impl VecZnxBig<FFT64> {
    pub fn print(&self, n: usize) {
        (0..self.size()).for_each(|i| println!("{}: {:?}", i, &self.at_limb(i)[..n]));
    }
}
