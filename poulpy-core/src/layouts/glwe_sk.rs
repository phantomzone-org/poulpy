use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, ReaderFrom, ScalarZnx, ScalarZnxToMut, ScalarZnxToRef, WriterTo, ZnxInfos, ZnxZero},
    source::Source,
};

use crate::{
    GetDistribution,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, LWEInfos, Rank, TorusPrecision},
};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWESecretLayout {
    pub n: Degree,
    pub rank: Rank,
}

impl LWEInfos for GLWESecretLayout {
    fn base2k(&self) -> Base2K {
        Base2K(0)
    }

    fn k(&self) -> TorusPrecision {
        TorusPrecision(0)
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn size(&self) -> usize {
        1
    }
}
impl GLWEInfos for GLWESecretLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GLWESecret<D: Data> {
    pub(crate) data: ScalarZnx<D>,
    pub(crate) dist: Distribution,
}

impl<D: Data> LWEInfos for GLWESecret<D> {
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

impl<D: Data> GetDistribution for GLWESecret<D> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data> GLWEInfos for GLWESecret<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32)
    }
}

impl GLWESecret<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.rank())
    }

    pub fn alloc(n: Degree, rank: Rank) -> Self {
        GLWESecret {
            data: ScalarZnx::alloc(n.into(), rank.into()),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.rank())
    }

    pub fn bytes_of(n: Degree, rank: Rank) -> usize {
        ScalarZnx::bytes_of(n.into(), rank.into())
    }
}

impl<D: DataMut> GLWESecret<D> {
    pub fn fill_ternary_prob(&mut self, prob: f64, source: &mut Source) {
        (0..self.rank().into()).for_each(|i| {
            self.data.fill_ternary_prob(i, prob, source);
        });
        self.dist = Distribution::TernaryProb(prob);
    }

    pub fn fill_ternary_hw(&mut self, hw: usize, source: &mut Source) {
        (0..self.rank().into()).for_each(|i| {
            self.data.fill_ternary_hw(i, hw, source);
        });
        self.dist = Distribution::TernaryFixed(hw);
    }

    pub fn fill_binary_prob(&mut self, prob: f64, source: &mut Source) {
        (0..self.rank().into()).for_each(|i| {
            self.data.fill_binary_prob(i, prob, source);
        });
        self.dist = Distribution::BinaryProb(prob);
    }

    pub fn fill_binary_hw(&mut self, hw: usize, source: &mut Source) {
        (0..self.rank().into()).for_each(|i| {
            self.data.fill_binary_hw(i, hw, source);
        });
        self.dist = Distribution::BinaryFixed(hw);
    }

    pub fn fill_binary_block(&mut self, block_size: usize, source: &mut Source) {
        (0..self.rank().into()).for_each(|i| {
            self.data.fill_binary_block(i, block_size, source);
        });
        self.dist = Distribution::BinaryBlock(block_size);
    }

    pub fn fill_zero(&mut self) {
        self.data.zero();
        self.dist = Distribution::ZERO;
    }
}

pub trait GLWESecretToMut {
    fn to_mut(&mut self) -> GLWESecret<&mut [u8]>;
}

impl<D: DataMut> GLWESecretToMut for GLWESecret<D> {
    fn to_mut(&mut self) -> GLWESecret<&mut [u8]> {
        GLWESecret {
            dist: self.dist,
            data: self.data.to_mut(),
        }
    }
}

pub trait GLWESecretToRef {
    fn to_ref(&self) -> GLWESecret<&[u8]>;
}

impl<D: DataRef> GLWESecretToRef for GLWESecret<D> {
    fn to_ref(&self) -> GLWESecret<&[u8]> {
        GLWESecret {
            data: self.data.to_ref(),
            dist: self.dist,
        }
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
