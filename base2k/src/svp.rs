use crate::ffi::svp::{self, svp_ppol_t};
use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::{assert_alignement, Module, VecZnx, VecZnxDft, BACKEND, LAYOUT};

use crate::{Infos, alloc_aligned, cast_mut};
use rand::seq::SliceRandom;
use rand_core::RngCore;
use rand_distr::{Distribution, weighted::WeightedIndex};
use sampling::source::Source;

pub struct Scalar {
    pub n: usize,
    pub data: Vec<i64>,
    pub ptr: *mut i64,
}

impl Module {
    pub fn new_scalar(&self) -> Scalar {
        Scalar::new(self.n())
    }
}

impl Scalar {
    pub fn new(n: usize) -> Self {
        let mut data: Vec<i64> = alloc_aligned::<i64>(n);
        let ptr: *mut i64 = data.as_mut_ptr();
        Self {
            n: n,
            data: data,
            ptr: ptr,
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn bytes_of(n: usize) -> usize {
        n * std::mem::size_of::<i64>()
    }

    pub fn from_bytes(n: usize, bytes: &mut [u8]) -> Self {
        let size: usize = Self::bytes_of(n);
        debug_assert!(
            bytes.len() == size,
            "invalid buffer: bytes.len()={} < self.bytes_of(n={})={}",
            bytes.len(),
            n,
            size
        );
        #[cfg(debug_assertions)]
        {
            assert_alignement(bytes.as_ptr())
        }
        unsafe {
            let bytes_i64: &mut [i64] = cast_mut::<u8, i64>(bytes);
            let ptr: *mut i64 = bytes_i64.as_mut_ptr();
            Self {
                n: n,
                data: Vec::from_raw_parts(bytes_i64.as_mut_ptr(), bytes.len(), bytes.len()),
                ptr: ptr,
            }
        }
    }

    pub fn from_bytes_borrow(n: usize, bytes: &mut [u8]) -> Self {
        let size: usize = Self::bytes_of(n);
        debug_assert!(
            bytes.len() == size,
            "invalid buffer: bytes.len()={} < self.bytes_of(n={})={}",
            bytes.len(),
            n,
            size
        );
        #[cfg(debug_assertions)]
        {
            assert_alignement(bytes.as_ptr())
        }
        let bytes_i64: &mut [i64] = cast_mut::<u8, i64>(bytes);
        let ptr: *mut i64 = bytes_i64.as_mut_ptr();
        Self {
            n: n,
            data: Vec::new(),
            ptr: ptr,
        }
    }

    pub fn as_ptr(&self) -> *const i64 {
        self.ptr
    }

    pub fn raw(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.n) }
    }

    pub fn raw_mut(&self) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.n) }
    }

    pub fn fill_ternary_prob(&mut self, prob: f64, source: &mut Source) {
        let choices: [i64; 3] = [-1, 0, 1];
        let weights: [f64; 3] = [prob / 2.0, 1.0 - prob, prob / 2.0];
        let dist: WeightedIndex<f64> = WeightedIndex::new(&weights).unwrap();
        self.data
            .iter_mut()
            .for_each(|x: &mut i64| *x = choices[dist.sample(source)]);
    }

    pub fn fill_ternary_hw(&mut self, hw: usize, source: &mut Source) {
        assert!(hw <= self.n());
        self.data[..hw]
            .iter_mut()
            .for_each(|x: &mut i64| *x = (((source.next_u32() & 1) as i64) << 1) - 1);
        self.data.shuffle(source);
    }

    pub fn as_vec_znx(&self) -> VecZnx {
        VecZnx {
            n: self.n,
            size: 1, // TODO REVIEW IF NEED TO ADD size TO SCALAR
            cols: 1,
            layout: LAYOUT::COL,
            data: Vec::new(),
            ptr: self.ptr,
        }
    }
}

pub trait ScalarOps {
    fn bytes_of_scalar(&self) -> usize;
    fn new_scalar(&self) -> Scalar;
    fn new_scalar_from_bytes(&self, bytes: &mut [u8]) -> Scalar;
    fn new_scalar_from_bytes_borrow(&self, tmp_bytes: &mut [u8]) -> Scalar;
}
impl ScalarOps for Module {
    fn bytes_of_scalar(&self) -> usize {
        Scalar::bytes_of(self.n())
    }
    fn new_scalar(&self) -> Scalar {
        Scalar::new(self.n())
    }
    fn new_scalar_from_bytes(&self, bytes: &mut [u8]) -> Scalar {
        Scalar::from_bytes(self.n(), bytes)
    }
    fn new_scalar_from_bytes_borrow(&self, tmp_bytes: &mut [u8]) -> Scalar {
        Scalar::from_bytes_borrow(self.n(), tmp_bytes)
    }
}

pub struct SvpPPol {
    pub n: usize,
    pub data: Vec<u8>,
    pub ptr: *mut u8,
    pub backend: BACKEND,
}

/// A prepared [crate::Scalar] for [SvpPPolOps::svp_apply_dft].
/// An [SvpPPol] an be seen as a [VecZnxDft] of one limb.
impl SvpPPol {
    pub fn new(module: &Module) -> Self {
        module.new_svp_ppol()
    }

    /// Returns the ring degree of the [SvpPPol].
    pub fn n(&self) -> usize {
        self.n
    }

    pub fn bytes_of(module: &Module) -> usize {
        module.bytes_of_svp_ppol()
    }

    pub fn from_bytes(module: &Module, bytes: &mut [u8]) -> SvpPPol {
        #[cfg(debug_assertions)]
        {
            assert_alignement(bytes.as_ptr());
            assert_eq!(bytes.len(), module.bytes_of_svp_ppol());
        }
        unsafe {
            Self {
                n: module.n(),
                data: Vec::from_raw_parts(bytes.as_mut_ptr(), bytes.len(), bytes.len()),
                ptr: bytes.as_mut_ptr(),
                backend: module.backend(),
            }
        }
    }

    pub fn from_bytes_borrow(module: &Module, tmp_bytes: &mut [u8]) -> SvpPPol {
        #[cfg(debug_assertions)]
        {
            assert_alignement(tmp_bytes.as_ptr());
            assert_eq!(tmp_bytes.len(), module.bytes_of_svp_ppol());
        }
        Self {
            n: module.n(),
            data: Vec::new(),
            ptr: tmp_bytes.as_mut_ptr(),
            backend: module.backend(),
        }
    }

    /// Returns the number of cols of the [SvpPPol], which is always 1.
    pub fn cols(&self) -> usize {
        1
    }
}

pub trait SvpPPolOps {
    /// Allocates a new [SvpPPol].
    fn new_svp_ppol(&self) -> SvpPPol;

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [SvpPPol] through [SvpPPol::from_bytes] ro.
    fn bytes_of_svp_ppol(&self) -> usize;

    /// Allocates a new [SvpPPol] from an array of bytes.
    /// The array of bytes is owned by the [SvpPPol].
    /// The method will panic if bytes.len() < [SvpPPolOps::bytes_of_svp_ppol]
    fn new_svp_ppol_from_bytes(&self, bytes: &mut [u8]) -> SvpPPol;

    /// Allocates a new [SvpPPol] from an array of bytes.
    /// The array of bytes is borrowed by the [SvpPPol].
    /// The method will panic if bytes.len() < [SvpPPolOps::bytes_of_svp_ppol]
    fn new_svp_ppol_from_bytes_borrow(&self, tmp_bytes: &mut [u8]) -> SvpPPol;

    /// Prepares a [crate::Scalar] for a [SvpPPolOps::svp_apply_dft].
    fn svp_prepare(&self, svp_ppol: &mut SvpPPol, a: &Scalar);

    /// Applies the [SvpPPol] x [VecZnxDft] product, where each limb of
    /// the [VecZnxDft] is multiplied with [SvpPPol].
    fn svp_apply_dft(&self, c: &mut VecZnxDft, a: &SvpPPol, b: &VecZnx);
}

impl SvpPPolOps for Module {
    fn new_svp_ppol(&self) -> SvpPPol {
        let mut data: Vec<u8> = alloc_aligned::<u8>(self.bytes_of_svp_ppol());
        let ptr: *mut u8 = data.as_mut_ptr();
        SvpPPol {
            data: data,
            ptr: ptr,
            n: self.n(),
            backend: self.backend(),
        }
    }

    fn bytes_of_svp_ppol(&self) -> usize {
        unsafe { svp::bytes_of_svp_ppol(self.ptr) as usize }
    }

    fn new_svp_ppol_from_bytes(&self, bytes: &mut [u8]) -> SvpPPol {
        SvpPPol::from_bytes(self, bytes)
    }

    fn new_svp_ppol_from_bytes_borrow(&self, tmp_bytes: &mut [u8]) -> SvpPPol {
        SvpPPol::from_bytes_borrow(self, tmp_bytes)
    }

    fn svp_prepare(&self, svp_ppol: &mut SvpPPol, a: &Scalar) {
        unsafe { svp::svp_prepare(self.ptr, svp_ppol.ptr as *mut svp_ppol_t, a.as_ptr()) }
    }

    fn svp_apply_dft(&self, c: &mut VecZnxDft, a: &SvpPPol, b: &VecZnx) {
        unsafe {
            svp::svp_apply_dft(
                self.ptr,
                c.ptr as *mut vec_znx_dft_t,
                c.cols() as u64,
                a.ptr as *const svp_ppol_t,
                b.as_ptr(),
                b.cols() as u64,
                b.n() as u64,
            )
        }
    }
}
