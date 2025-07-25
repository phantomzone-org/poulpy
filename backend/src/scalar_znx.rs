use crate::ffi::vec_znx;
use crate::znx_base::ZnxInfos;
use crate::{
    Backend, DataView, DataViewMut, Module, VecZnx, VecZnxToMut, VecZnxToRef, ZnxSliceSize, ZnxView, ZnxViewMut, alloc_aligned,
};
use rand::seq::SliceRandom;
use rand_core::RngCore;
use rand_distr::{Distribution, weighted::WeightedIndex};
use sampling::source::Source;

pub struct ScalarZnx<D> {
    pub(crate) data: D,
    pub(crate) n: usize,
    pub(crate) cols: usize,
}

impl<D> ZnxInfos for ScalarZnx<D> {
    fn cols(&self) -> usize {
        self.cols
    }

    fn rows(&self) -> usize {
        1
    }

    fn n(&self) -> usize {
        self.n
    }

    fn size(&self) -> usize {
        1
    }
}

impl<D> ZnxSliceSize for ScalarZnx<D> {
    fn sl(&self) -> usize {
        self.n()
    }
}

impl<D> DataView for ScalarZnx<D> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D> DataViewMut for ScalarZnx<D> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: AsRef<[u8]>> ZnxView for ScalarZnx<D> {
    type Scalar = i64;
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> ScalarZnx<D> {
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

    pub fn fill_binary_prob(&mut self, col: usize, prob: f64, source: &mut Source) {
        let choices: [i64; 2] = [0, 1];
        let weights: [f64; 2] = [1.0 - prob, prob];
        let dist: WeightedIndex<f64> = WeightedIndex::new(&weights).unwrap();
        self.at_mut(col, 0)
            .iter_mut()
            .for_each(|x: &mut i64| *x = choices[dist.sample(source)]);
    }

    pub fn fill_binary_hw(&mut self, col: usize, hw: usize, source: &mut Source) {
        assert!(hw <= self.n());
        self.at_mut(col, 0)[..hw]
            .iter_mut()
            .for_each(|x: &mut i64| *x = (source.next_u32() & 1) as i64);
        self.at_mut(col, 0).shuffle(source);
    }

    pub fn fill_binary_block(&mut self, col: usize, block_size: usize, source: &mut Source) {
        assert!(self.n() % block_size == 0);
        let max_idx: u64 = (block_size + 1) as u64;
        let mask_idx: u64 = (1 << ((u64::BITS - max_idx.leading_zeros()) as u64)) - 1;
        for block in self.at_mut(col, 0).chunks_mut(block_size) {
            let idx: usize = source.next_u64n(max_idx, mask_idx) as usize;
            if idx != block_size {
                block[idx] = 1;
            }
        }
    }
}

impl<D: From<Vec<u8>>> ScalarZnx<D> {
    pub(crate) fn bytes_of(n: usize, cols: usize) -> usize {
        n * cols * size_of::<i64>()
    }

    pub fn new(n: usize, cols: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of(n, cols));
        Self {
            data: data.into(),
            n,
            cols,
        }
    }

    pub(crate) fn new_from_bytes(n: usize, cols: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(n, cols));
        Self {
            data: data.into(),
            n,
            cols,
        }
    }
}

pub type ScalarZnxOwned = ScalarZnx<Vec<u8>>;

pub(crate) fn bytes_of_scalar_znx<B: Backend>(module: &Module<B>, cols: usize) -> usize {
    ScalarZnxOwned::bytes_of(module.n(), cols)
}

pub trait ScalarZnxAlloc {
    fn bytes_of_scalar_znx(&self, cols: usize) -> usize;
    fn new_scalar_znx(&self, cols: usize) -> ScalarZnxOwned;
    fn new_scalar_znx_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> ScalarZnxOwned;
}

impl<B: Backend> ScalarZnxAlloc for Module<B> {
    fn bytes_of_scalar_znx(&self, cols: usize) -> usize {
        ScalarZnxOwned::bytes_of(self.n(), cols)
    }
    fn new_scalar_znx(&self, cols: usize) -> ScalarZnxOwned {
        ScalarZnxOwned::new(self.n(), cols)
    }
    fn new_scalar_znx_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> ScalarZnxOwned {
        ScalarZnxOwned::new_from_bytes(self.n(), cols, bytes)
    }
}

pub trait ScalarZnxOps {
    fn scalar_znx_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxToMut,
        A: ScalarZnxToRef;

    /// Applies the automorphism X^i -> X^ik on the selected column of `a`.
    fn scalar_znx_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: ScalarZnxToMut;
}

impl<B: Backend> ScalarZnxOps for Module<B> {
    fn scalar_znx_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxToMut,
        A: ScalarZnxToRef,
    {
        let a: ScalarZnx<&[u8]> = a.to_ref();
        let mut res: ScalarZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.ptr,
                k,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }

    fn scalar_znx_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: ScalarZnxToMut,
    {
        let mut a: ScalarZnx<&mut [u8]> = a.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.ptr,
                k,
                a.at_mut_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }
}

impl<D> ScalarZnx<D> {
    pub(crate) fn from_data(data: D, n: usize, cols: usize) -> Self {
        Self { data, n, cols }
    }
}

pub trait ScalarZnxToRef {
    fn to_ref(&self) -> ScalarZnx<&[u8]>;
}

impl<D> ScalarZnxToRef for ScalarZnx<D>
where
    D: AsRef<[u8]>,
{
    fn to_ref(&self) -> ScalarZnx<&[u8]> {
        ScalarZnx {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
        }
    }
}

pub trait ScalarZnxToMut {
    fn to_mut(&mut self) -> ScalarZnx<&mut [u8]>;
}

impl<D> ScalarZnxToMut for ScalarZnx<D>
where
    D: AsRef<[u8]> + AsMut<[u8]>,
{
    fn to_mut(&mut self) -> ScalarZnx<&mut [u8]> {
        ScalarZnx {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
        }
    }
}

impl<D> VecZnxToRef for ScalarZnx<D>
where
    D: AsRef<[u8]>,
{
    fn to_ref(&self) -> VecZnx<&[u8]> {
        VecZnx {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            size: 1,
        }
    }
}

impl<D> VecZnxToMut for ScalarZnx<D>
where
    D: AsRef<[u8]> + AsMut<[u8]>,
{
    fn to_mut(&mut self) -> VecZnx<&mut [u8]> {
        VecZnx {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            size: 1,
        }
    }
}
