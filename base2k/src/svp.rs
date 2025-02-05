use crate::ffi::svp;
use crate::{Module, VecZnx, VecZnxDft};

use crate::Infos;
use rand::seq::SliceRandom;
use rand_core::RngCore;
use rand_distr::{Distribution, WeightedIndex};
use sampling::source::Source;

pub struct Scalar(pub Vec<i64>);

impl Module {
    pub fn new_scalar(&self) -> Scalar {
        Scalar::new(self.n())
    }
}

impl Scalar {
    pub fn new(n: usize) -> Self {
        Self(vec![i64::default(); Self::buffer_size(n)])
    }

    pub fn buffer_size(n: usize) -> usize {
        n
    }

    pub fn from_buffer(&mut self, n: usize, buf: &[i64]) {
        let size: usize = Self::buffer_size(n);
        assert!(
            buf.len() >= size,
            "invalid buffer: buf.len()={} < self.buffer_size(n={})={}",
            buf.len(),
            n,
            size
        );
        self.0 = Vec::from(&buf[..size])
    }

    pub fn as_ptr(&self) -> *const i64 {
        self.0.as_ptr()
    }

    pub fn fill_ternary_prob(&mut self, prob: f64, source: &mut Source) {
        let choices: [i64; 3] = [-1, 0, 1];
        let weights: [f64; 3] = [prob / 2.0, 1.0 - prob, prob / 2.0];
        let dist: WeightedIndex<f64> = WeightedIndex::new(&weights).unwrap();
        self.0
            .iter_mut()
            .for_each(|x: &mut i64| *x = choices[dist.sample(source)]);
    }

    pub fn fill_ternary_hw(&mut self, hw: usize, source: &mut Source) {
        self.0[..hw]
            .iter_mut()
            .for_each(|x: &mut i64| *x = (((source.next_u32() & 1) as i64) << 1) - 1);
        self.0.shuffle(source);
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

    /// Returns the number of limbs of the [SvpPPol], which is always 1.
    pub fn limbs(&self) -> usize {
        1
    }
}

pub trait SvpPPolOps {
    /// Prepares a [crate::Scalar] for a [SvpPPolOps::svp_apply_dft].
    fn svp_prepare(&self, svp_ppol: &mut SvpPPol, a: &Scalar);

    /// Allocates a new [SvpPPol].
    fn svp_new_ppol(&self) -> SvpPPol;

    /// Applies the [SvpPPol] x [VecZnxDft] product, where each limb of
    /// the [VecZnxDft] is multiplied with [SvpPPol].
    fn svp_apply_dft(&self, c: &mut VecZnxDft, a: &SvpPPol, b: &VecZnx);
}

impl SvpPPolOps for Module {
    fn svp_prepare(&self, svp_ppol: &mut SvpPPol, a: &Scalar) {
        unsafe { svp::svp_prepare(self.0, svp_ppol.0, a.as_ptr()) }
    }

    fn svp_new_ppol(&self) -> SvpPPol {
        unsafe { SvpPPol(svp::new_svp_ppol(self.0), self.n()) }
    }

    fn svp_apply_dft(&self, c: &mut VecZnxDft, a: &SvpPPol, b: &VecZnx) {
        let limbs: u64 = b.limbs() as u64;
        assert!(
            c.limbs() as u64 >= limbs,
            "invalid c_vector: c_vector.limbs()={} < b.limbs()={}",
            c.limbs(),
            limbs
        );
        unsafe { svp::svp_apply_dft(self.0, c.0, limbs, a.0, b.as_ptr(), limbs, b.n() as u64) }
    }
}
