use crate::ffi::svp;
use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::{assert_alignement, Module, VecZnx, VecZnxDft};

use crate::{alloc_aligned, cast_mut, Infos};
use rand::seq::SliceRandom;
use rand_core::RngCore;
use rand_distr::{Distribution, WeightedIndex};
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

    pub fn buffer_size(n: usize) -> usize {
        n
    }

    pub fn from_buffer(&mut self, n: usize, bytes: &mut [u8]) -> Self {
        let size: usize = Self::buffer_size(n);
        debug_assert!(
            bytes.len() == size,
            "invalid buffer: bytes.len()={} < self.buffer_size(n={})={}",
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

    pub fn as_ptr(&self) -> *const i64 {
        self.ptr
    }

    pub fn raw(&self) -> &[i64] {
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
}

pub struct SvpPPol(pub *mut svp::svp_ppol_t, pub usize);

/// A prepared [crate::Scalar] for [SvpPPolOps::svp_apply_dft].
/// An [SvpPPol] an be seen as a [VecZnxDft] of one limb.
/// The backend array of an [SvpPPol] is allocated in C and must be freed manually.
impl SvpPPol {
    /// Returns the ring degree of the [SvpPPol].
    pub fn n(&self) -> usize {
        self.1
    }

    pub fn from_bytes(size: usize, bytes: &mut [u8]) -> SvpPPol {
        #[cfg(debug_assertions)]
        {
            assert_alignement(bytes.as_ptr())
        }
        debug_assert!(bytes.len() << 3 >= size);
        SvpPPol(bytes.as_mut_ptr() as *mut svp::svp_ppol_t, size)
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
    /// a new [SvpPPol] through [SvpPPol::from_bytes].
    fn bytes_of_svp_ppol(&self) -> usize;

    /// Prepares a [crate::Scalar] for a [SvpPPolOps::svp_apply_dft].
    fn svp_prepare(&self, svp_ppol: &mut SvpPPol, a: &Scalar);

    /// Applies the [SvpPPol] x [VecZnxDft] product, where each limb of
    /// the [VecZnxDft] is multiplied with [SvpPPol].
    fn svp_apply_dft(&self, c: &mut VecZnxDft, a: &SvpPPol, b: &VecZnx, b_cols: usize);
}

impl SvpPPolOps for Module {
    fn new_svp_ppol(&self) -> SvpPPol {
        unsafe { SvpPPol(svp::new_svp_ppol(self.ptr), self.n()) }
    }

    fn bytes_of_svp_ppol(&self) -> usize {
        unsafe { svp::bytes_of_svp_ppol(self.ptr) as usize }
    }

    fn svp_prepare(&self, svp_ppol: &mut SvpPPol, a: &Scalar) {
        unsafe { svp::svp_prepare(self.ptr, svp_ppol.0, a.as_ptr()) }
    }

    fn svp_apply_dft(&self, c: &mut VecZnxDft, a: &SvpPPol, b: &VecZnx, b_cols: usize) {
        debug_assert!(
            c.cols() >= b_cols,
            "invalid c_vector: c_vector.cols()={} < b.cols()={}",
            c.cols(),
            b_cols
        );
        unsafe {
            svp::svp_apply_dft(
                self.ptr,
                c.ptr as *mut vec_znx_dft_t,
                b_cols as u64,
                a.0,
                b.as_ptr(),
                b_cols as u64,
                b.n() as u64,
            )
        }
    }
}
