use backend::hal::{
    api::{ZnxInfos, ZnxZero},
    layouts::ScalarZnx,
};
use sampling::source::Source;

use crate::Distribution;

pub struct LWESecret<T> {
    pub(crate) data: ScalarZnx<T>,
    pub(crate) dist: Distribution,
}

impl LWESecret<Vec<u8>> {
    pub fn alloc(n: usize) -> Self {
        Self {
            data: ScalarZnx::new(n, 1),
            dist: Distribution::NONE,
        }
    }
}

impl<DataSelf> LWESecret<DataSelf> {
    pub fn n(&self) -> usize {
        self.data.n()
    }

    pub fn log_n(&self) -> usize {
        self.data.log_n()
    }

    pub fn rank(&self) -> usize {
        self.data.cols()
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>> LWESecret<D> {
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
