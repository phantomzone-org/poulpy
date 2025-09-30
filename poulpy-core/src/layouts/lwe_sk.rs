use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, ScalarZnx, ZnxInfos, ZnxView, ZnxZero},
    source::Source,
};

use crate::{
    dist::Distribution,
    layouts::{Base2K, Degree, LWEInfos, TorusPrecision},
};

pub struct LWESecret<D: Data> {
    pub(crate) data: ScalarZnx<D>,
    pub(crate) dist: Distribution,
}

impl LWESecret<Vec<u8>> {
    pub fn alloc(n: Degree) -> Self {
        Self {
            data: ScalarZnx::alloc(n.into(), 1),
            dist: Distribution::NONE,
        }
    }
}

impl<D: DataRef> LWESecret<D> {
    pub fn raw(&self) -> &[i64] {
        self.data.at(0, 0)
    }

    pub fn dist(&self) -> Distribution {
        self.dist
    }

    pub fn data(&self) -> &ScalarZnx<D> {
        &self.data
    }
}

impl<D: Data> LWEInfos for LWESecret<D> {
    fn base2k(&self) -> Base2K {
        Base2K(0)
    }
    fn k(&self) -> TorusPrecision {
        TorusPrecision(0)
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
        1
    }
}

impl<D: DataMut> LWESecret<D> {
    pub fn fill_ternary_prob(&mut self, prob: f64, source: &mut Source) {
        self.data.fill_ternary_prob(0, prob, source);
        self.dist = Distribution::TernaryProb(prob);
    }

    pub fn fill_ternary_hw(&mut self, hw: usize, source: &mut Source) {
        self.data.fill_ternary_hw(0, hw, source);
        self.dist = Distribution::TernaryFixed(hw);
    }

    pub fn fill_binary_prob(&mut self, prob: f64, source: &mut Source) {
        self.data.fill_binary_prob(0, prob, source);
        self.dist = Distribution::BinaryProb(prob);
    }

    pub fn fill_binary_hw(&mut self, hw: usize, source: &mut Source) {
        self.data.fill_binary_hw(0, hw, source);
        self.dist = Distribution::BinaryFixed(hw);
    }

    pub fn fill_binary_block(&mut self, block_size: usize, source: &mut Source) {
        self.data.fill_binary_block(0, block_size, source);
        self.dist = Distribution::BinaryBlock(block_size);
    }

    pub fn fill_zero(&mut self) {
        self.data.zero();
        self.dist = Distribution::ZERO;
    }
}
