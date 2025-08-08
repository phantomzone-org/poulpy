use backend::hal::{
    api::{ScalarZnxAlloc, ScalarZnxAllocBytes, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, ZnxInfos, ZnxZero},
    layouts::{Backend, Data, DataMut, DataRef, Module, ReaderFrom, ScalarZnx, SvpPPol, WriterTo},
};
use sampling::source::Source;

use crate::dist::Distribution;

pub trait GLWESecretFamily<B: Backend> = SvpPrepare<B> + SvpPPolAllocBytes + SvpPPolAlloc<B>;

#[derive(PartialEq, Eq)]
pub struct GLWESecret<D: Data> {
    pub(crate) data: ScalarZnx<D>,
    pub(crate) dist: Distribution,
}

impl GLWESecret<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, rank: usize) -> Self
    where
        Module<B>: ScalarZnxAlloc,
    {
        Self {
            data: module.scalar_znx_alloc(rank),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, rank: usize) -> usize
    where
        Module<B>: ScalarZnxAllocBytes,
    {
        module.scalar_znx_alloc_bytes(rank)
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

pub struct GLWESecretExec<D: Data, B: Backend> {
    pub(crate) data: SvpPPol<D, B>,
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
        D: DataRef,
        Module<B>: GLWESecretFamily<B>,
    {
        let mut sk_dft: GLWESecretExec<Vec<u8>, B> = Self::alloc(module, sk.rank());
        sk_dft.prepare(module, sk);
        sk_dft
    }
}

impl<D: Data, B: Backend> GLWESecretExec<D, B> {
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

impl<D: DataMut, B: Backend> GLWESecretExec<D, B> {
    pub(crate) fn prepare<O>(&mut self, module: &Module<B>, sk: &GLWESecret<O>)
    where
        O: DataRef,
        Module<B>: GLWESecretFamily<B>,
    {
        (0..self.rank()).for_each(|i| {
            module.svp_prepare(&mut self.data, i, &sk.data, i);
        });
        self.dist = sk.dist
    }
}
