use backend::hal::{
    api::{ZnxInfos, ZnxZero},
    layouts::{Data, DataMut, DataRef, ReaderFrom, ScalarZnx, WriterTo},
};
use sampling::source::Source;

use crate::dist::Distribution;

#[derive(PartialEq, Eq, Clone)]
pub struct GLWESecret<D: Data> {
    pub(crate) data: ScalarZnx<D>,
    pub(crate) dist: Distribution,
}

impl GLWESecret<Vec<u8>> {
    pub fn alloc(n: usize, rank: usize) -> Self {
        Self {
            data: ScalarZnx::alloc(n, rank),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of(n: usize, rank: usize) -> usize {
        ScalarZnx::alloc_bytes(n, rank)
    }
}

impl<D: Data> GLWESecret<D> {
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

impl<D: DataMut> GLWESecret<D> {
    pub fn fill_ternary_prob(&mut self, prob: f64, source: &mut Source) {
        (0..self.rank()).for_each(|i| {
            self.data.fill_ternary_prob(i, prob, source);
        });
        self.dist = Distribution::TernaryProb(prob);
    }

    pub fn fill_ternary_hw(&mut self, hw: usize, source: &mut Source) {
        (0..self.rank()).for_each(|i| {
            self.data.fill_ternary_hw(i, hw, source);
        });
        self.dist = Distribution::TernaryFixed(hw);
    }

    pub fn fill_binary_prob(&mut self, prob: f64, source: &mut Source) {
        (0..self.rank()).for_each(|i| {
            self.data.fill_binary_prob(i, prob, source);
        });
        self.dist = Distribution::BinaryProb(prob);
    }

    pub fn fill_binary_hw(&mut self, hw: usize, source: &mut Source) {
        (0..self.rank()).for_each(|i| {
            self.data.fill_binary_hw(i, hw, source);
        });
        self.dist = Distribution::BinaryFixed(hw);
    }

    pub fn fill_binary_block(&mut self, block_size: usize, source: &mut Source) {
        (0..self.rank()).for_each(|i| {
            self.data.fill_binary_block(i, block_size, source);
        });
        self.dist = Distribution::BinaryBlock(block_size);
    }

    pub fn fill_zero(&mut self) {
        self.data.zero();
        self.dist = Distribution::ZERO;
    }
}

impl<D: DataMut> ReaderFrom for GLWESecret<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        match Distribution::read_from(reader) {
            Ok(dist) => self.dist = dist,
            Err(e) => return Err(e),
        }
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWESecret<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        match self.dist.write_to(writer) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        self.data.write_to(writer)
    }
}
