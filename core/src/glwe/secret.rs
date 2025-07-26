use backend::{
    Backend, Module, ScalarZnx, ScalarZnxAlloc, SvpPPol, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, ZnxInfos, ZnxZero,
};
use sampling::source::Source;

use crate::dist::Distribution;

pub trait GLWESecretFamily<B: Backend> = SvpPrepare<B> + SvpPPolAllocBytes + SvpPPolAlloc<B>;

pub struct GLWESecret<T> {
    pub(crate) data: ScalarZnx<T>,
    pub(crate) dist: Distribution,
}

impl GLWESecret<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, rank: usize) -> Self {
        Self {
            data: module.new_scalar_znx(rank),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, rank: usize) -> usize {
        module.bytes_of_scalar_znx(rank)
    }
}

impl<DataSelf> GLWESecret<DataSelf> {
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

impl<S: AsMut<[u8]> + AsRef<[u8]>> GLWESecret<S> {
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

pub struct GLWESecretExec<T, B: Backend> {
    pub(crate) data: SvpPPol<T, B>,
    pub(crate) dist: Distribution,
}

impl<B: Backend> GLWESecretExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, rank: usize) -> Self
    where
        Module<B>: GLWESecretFamily<B>,
    {
        Self {
            data: module.svp_ppol_alloc(rank),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<B>, rank: usize) -> usize
    where
        Module<B>: GLWESecretFamily<B>,
    {
        module.svp_ppol_alloc_bytes(rank)
    }
}

impl<B: Backend> GLWESecretExec<Vec<u8>, B> {
    pub fn from<D>(module: &Module<B>, sk: &GLWESecret<D>) -> Self
    where
        D: AsRef<[u8]>,
        Module<B>: GLWESecretFamily<B>,
    {
        let mut sk_dft: GLWESecretExec<Vec<u8>, B> = Self::alloc(module, sk.rank());
        sk_dft.prepare(module, sk);
        sk_dft
    }
}

impl<DataSelf, B: Backend> GLWESecretExec<DataSelf, B> {
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

impl<S: AsMut<[u8]> + AsRef<[u8]>, B: Backend> GLWESecretExec<S, B> {
    pub(crate) fn prepare<D>(&mut self, module: &Module<B>, sk: &GLWESecret<D>)
    where
        D: AsRef<[u8]>,
        Module<B>: GLWESecretFamily<B>,
    {
        (0..self.rank()).for_each(|i| {
            module.svp_prepare(&mut self.data, i, &sk.data, i);
        });
        self.dist = sk.dist
    }
}
