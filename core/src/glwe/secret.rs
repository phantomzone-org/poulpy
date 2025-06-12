use backend::{
    Backend, FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxDft, ScalarZnxDftAlloc, ScalarZnxDftOps, ZnxInfos, ZnxZero,
};
use sampling::source::Source;

use crate::keys::SecretDistribution;

pub struct GLWESecret<T, B: Backend> {
    pub(crate) data: ScalarZnx<T>,
    pub(crate) data_fourier: ScalarZnxDft<T, B>,
    pub(crate) dist: SecretDistribution,
}

impl<B: Backend> GLWESecret<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, rank: usize) -> Self {
        Self {
            data: module.new_scalar_znx(rank),
            data_fourier: module.new_scalar_znx_dft(rank),
            dist: SecretDistribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<B>, rank: usize) -> usize {
        module.bytes_of_scalar_znx(rank) + module.bytes_of_scalar_znx_dft(rank)
    }
}

impl<DataSelf, B: Backend> GLWESecret<DataSelf, B> {
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

impl<S: AsMut<[u8]> + AsRef<[u8]>> GLWESecret<S, FFT64> {
    pub fn fill_ternary_prob(&mut self, module: &Module<FFT64>, prob: f64, source: &mut Source) {
        (0..self.rank()).for_each(|i| {
            self.data.fill_ternary_prob(i, prob, source);
        });
        self.prep_fourier(module);
        self.dist = SecretDistribution::TernaryProb(prob);
    }

    pub fn fill_ternary_hw(&mut self, module: &Module<FFT64>, hw: usize, source: &mut Source) {
        (0..self.rank()).for_each(|i| {
            self.data.fill_ternary_hw(i, hw, source);
        });
        self.prep_fourier(module);
        self.dist = SecretDistribution::TernaryFixed(hw);
    }

    pub fn fill_binary_prob(&mut self, module: &Module<FFT64>, prob: f64, source: &mut Source) {
        (0..self.rank()).for_each(|i| {
            self.data.fill_binary_prob(i, prob, source);
        });
        self.prep_fourier(module);
        self.dist = SecretDistribution::BinaryProb(prob);
    }

    pub fn fill_binary_hw(&mut self, module: &Module<FFT64>, hw: usize, source: &mut Source) {
        (0..self.rank()).for_each(|i| {
            self.data.fill_binary_hw(i, hw, source);
        });
        self.prep_fourier(module);
        self.dist = SecretDistribution::BinaryFixed(hw);
    }

    pub fn fill_binary_block(&mut self, module: &Module<FFT64>, block_size: usize, source: &mut Source) {
        (0..self.rank()).for_each(|i| {
            self.data.fill_binary_block(i, block_size, source);
        });
        self.prep_fourier(module);
        self.dist = SecretDistribution::BinaryBlock(block_size);
    }

    pub fn fill_zero(&mut self) {
        self.data.zero();
        self.dist = SecretDistribution::ZERO;
    }

    pub(crate) fn prep_fourier(&mut self, module: &Module<FFT64>) {
        (0..self.rank()).for_each(|i| {
            module.svp_prepare(&mut self.data_fourier, i, &self.data, i);
        });
    }
}
