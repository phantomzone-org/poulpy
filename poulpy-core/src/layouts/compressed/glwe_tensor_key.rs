use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWECompressed, GGLWECompressedSeedMut, GGLWECompressedToMut, GGLWECompressedToRef,
    GGLWEDecompress, GGLWEInfos, GGLWEToMut, GLWEInfos, GLWETensorKey, LWEInfos, Rank, TorusPrecision,
};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GLWETensorKeyCompressed<D: Data>(pub(crate) GGLWECompressed<D>);

impl<D: DataMut> GGLWECompressedSeedMut for GLWETensorKeyCompressed<D> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.0.seed
    }
}

impl<D: Data> LWEInfos for GLWETensorKeyCompressed<D> {
    fn n(&self) -> Degree {
        self.0.n()
    }

    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.0.k()
    }
    fn limbs(&self) -> usize {
        self.0.limbs()
    }
}
impl<D: Data> GLWEInfos for GLWETensorKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GLWETensorKeyCompressed<D> {
    fn rank_in(&self) -> Rank {
        self.rank_out()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

impl<D: DataRef> fmt::Debug for GLWETensorKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GLWETensorKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for GLWETensorKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GLWETensorKeyCompressed)",)?;
        write!(f, "{}", self.0)?;
        Ok(())
    }
}

impl GLWETensorKeyCompressed<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        GLWETensorKeyCompressed(GGLWECompressed::alloc(n, base2k, k, Rank(pairs), rank, dnum, dsize))
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        GGLWECompressed::bytes_of(n, base2k, k, Rank(pairs), dnum, dsize)
    }
}

impl<D: DataMut> ReaderFrom for GLWETensorKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)?;
        Ok(())
    }
}

impl<D: DataRef> WriterTo for GLWETensorKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)?;
        Ok(())
    }
}

pub trait GLWETensorKeyDecompress
where
    Self: GGLWEDecompress,
{
    fn decompress_tensor_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToMut,
        O: GGLWECompressedToRef,
    {
        self.decompress_gglwe(res, other);
    }
}

impl<B: Backend> GLWETensorKeyDecompress for Module<B> where Self: GGLWEDecompress {}

impl<D: DataMut> GLWETensorKey<D> {
    pub fn decompress<O, M>(&mut self, module: &M, other: &O)
    where
        O: GGLWECompressedToRef,
        M: GLWETensorKeyDecompress,
    {
        module.decompress_tensor_key(self, other);
    }
}

impl<D: DataMut> GGLWECompressedToMut for GLWETensorKeyCompressed<D>
where
    GGLWECompressed<D>: GGLWECompressedToMut,
{
    fn to_mut(&mut self) -> GGLWECompressed<&mut [u8]> {
        self.0.to_mut()
    }
}

impl<D: DataRef> GGLWECompressedToRef for GLWETensorKeyCompressed<D>
where
    GGLWECompressed<D>: GGLWECompressedToRef,
{
    fn to_ref(&self) -> GGLWECompressed<&[u8]> {
        self.0.to_ref()
    }
}
