use crate::znx_base::{ZnxAlloc, ZnxBase, ZnxInfos, ZnxLayout, ZnxSliceSize};
use crate::{Backend, GetZnxBase, Module, VecZnx};
use rand::seq::SliceRandom;
use rand_core::RngCore;
use rand_distr::{Distribution, weighted::WeightedIndex};
use sampling::source::Source;

pub const SCALAR_ZNX_ROWS: usize = 1;
pub const SCALAR_ZNX_SIZE: usize = 1;

pub struct Scalar {
    pub inner: ZnxBase,
}

impl GetZnxBase for Scalar {
    fn znx(&self) -> &ZnxBase {
        &self.inner
    }

    fn znx_mut(&mut self) -> &mut ZnxBase {
        &mut self.inner
    }
}

impl ZnxInfos for Scalar {}

impl<B: Backend> ZnxAlloc<B> for Scalar {
    type Scalar = i64;

    fn from_bytes_borrow(module: &Module<B>, _rows: usize, cols: usize, _size: usize, bytes: &mut [u8]) -> Self {
        Self {
            inner: ZnxBase::from_bytes_borrow(module.n(), SCALAR_ZNX_ROWS, cols, SCALAR_ZNX_SIZE, bytes),
        }
    }

    fn bytes_of(module: &Module<B>, _rows: usize, cols: usize, _size: usize) -> usize {
        debug_assert_eq!(
            _rows, SCALAR_ZNX_ROWS,
            "rows != {} not supported for Scalar",
            SCALAR_ZNX_ROWS
        );
        debug_assert_eq!(
            _size, SCALAR_ZNX_SIZE,
            "rows != {} not supported for Scalar",
            SCALAR_ZNX_SIZE
        );
        module.n() * cols * std::mem::size_of::<self::Scalar>()
    }
}

impl ZnxLayout for Scalar {
    type Scalar = i64;
}

impl ZnxSliceSize for Scalar {
    fn sl(&self) -> usize {
        self.n()
    }
}

impl Scalar {
    pub fn fill_ternary_prob(&mut self, col: usize, prob: f64, source: &mut Source) {
        let choices: [i64; 3] = [-1, 0, 1];
        let weights: [f64; 3] = [prob / 2.0, 1.0 - prob, prob / 2.0];
        let dist: WeightedIndex<f64> = WeightedIndex::new(&weights).unwrap();
        self.at_mut(col, 0)
            .iter_mut()
            .for_each(|x: &mut i64| *x = choices[dist.sample(source)]);
    }

    pub fn fill_ternary_hw(&mut self, col: usize, hw: usize, source: &mut Source) {
        assert!(hw <= self.n());
        self.at_mut(col, 0)[..hw]
            .iter_mut()
            .for_each(|x: &mut i64| *x = (((source.next_u32() & 1) as i64) << 1) - 1);
        self.at_mut(col, 0).shuffle(source);
    }

    pub fn alias_as_vec_znx(&self) -> VecZnx {
        VecZnx {
            inner: ZnxBase {
                n: self.n(),
                rows: 1,
                cols: 1,
                size: 1,
                data: Vec::new(),
                ptr: self.ptr() as *mut u8,
            },
        }
    }
}

pub trait ScalarOps {
    fn bytes_of_scalar(&self, cols: usize) -> usize;
    fn new_scalar(&self, cols: usize) -> Scalar;
    fn new_scalar_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> Scalar;
    fn new_scalar_from_bytes_borrow(&self, cols: usize, bytes: &mut [u8]) -> Scalar;
}

impl<B: Backend> ScalarOps for Module<B> {
    fn bytes_of_scalar(&self, cols: usize) -> usize {
        Scalar::bytes_of(self, SCALAR_ZNX_ROWS, cols, SCALAR_ZNX_SIZE)
    }
    fn new_scalar(&self, cols: usize) -> Scalar {
        Scalar::new(self, SCALAR_ZNX_ROWS, cols, SCALAR_ZNX_SIZE)
    }
    fn new_scalar_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> Scalar {
        Scalar::from_bytes(self, SCALAR_ZNX_ROWS, cols, SCALAR_ZNX_SIZE, bytes)
    }
    fn new_scalar_from_bytes_borrow(&self, cols: usize, bytes: &mut [u8]) -> Scalar {
        Scalar::from_bytes_borrow(self, SCALAR_ZNX_ROWS, cols, SCALAR_ZNX_SIZE, bytes)
    }
}
