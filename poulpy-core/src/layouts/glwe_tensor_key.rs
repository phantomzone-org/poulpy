use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEInfos, GGLWEToMut, GGLWEToRef, GLWEInfos, LWEInfos, Rank, TorusPrecision,
};

use std::fmt;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWETensorKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

#[derive(PartialEq, Eq, Clone)]
pub struct GLWETensorKey<D: Data>(pub(crate) GGLWE<D>);

impl<D: Data> LWEInfos for GLWETensorKey<D> {
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

impl<D: Data> GLWEInfos for GLWETensorKey<D> {
    fn rank(&self) -> Rank {
        self.0.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GLWETensorKey<D> {
    fn rank_in(&self) -> Rank {
        let rank_out: usize = self.rank_out().as_usize();
        let pairs: usize = (((rank_out + 1) * rank_out) >> 1).max(1);
        pairs.into()
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

impl LWEInfos for GLWETensorKeyLayout {
    fn n(&self) -> Degree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl GLWEInfos for GLWETensorKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for GLWETensorKeyLayout {
    fn rank_in(&self) -> Rank {
        let rank_out: usize = self.rank_out().as_usize();
        let pairs: usize = (((rank_out + 1) * rank_out) >> 1).max(1);
        pairs.into()
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn rank_out(&self) -> Rank {
        self.rank
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

impl<D: DataRef> fmt::Debug for GLWETensorKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GLWETensorKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source)
    }
}

impl<D: DataRef> fmt::Display for GLWETensorKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GLWETensorKey)",)?;
        write!(f, "{}", self.0)?;
        Ok(())
    }
}

impl GLWETensorKey<Vec<u8>> {
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
        let pairs: u32 = (((rank.0 + 1) * rank.0) >> 1).max(1);
        GLWETensorKey(GGLWE::alloc(n, base2k, k, Rank(pairs), rank, dnum, dsize))
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
        let pairs: u32 = (((rank.0 + 1) * rank.0) >> 1).max(1);
        GGLWE::bytes_of(n, base2k, k, Rank(pairs), rank, dnum, dsize)
    }
}

impl<D: DataMut> ReaderFrom for GLWETensorKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)?;
        Ok(())
    }
}

impl<D: DataRef> WriterTo for GLWETensorKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)?;
        Ok(())
    }
}

impl<D: DataRef> GGLWEToRef for GLWETensorKey<D>
where
    GGLWE<D>: GGLWEToRef,
{
    fn to_ref(&self) -> GGLWE<&[u8]> {
        self.0.to_ref()
    }
}

impl<D: DataMut> GGLWEToMut for GLWETensorKey<D>
where
    GGLWE<D>: GGLWEToMut,
{
    fn to_mut(&mut self) -> GGLWE<&mut [u8]> {
        self.0.to_mut()
    }
}
