use std::fmt;

use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision, compressed::GGLWESwitchingKeyCompressed,
};

#[derive(PartialEq, Eq, Clone)]
pub struct GLWEToLWESwitchingKeyCompressed<D: Data>(pub(crate) GGLWESwitchingKeyCompressed<D>);

impl<D: Data> LWEInfos for GLWEToLWESwitchingKeyCompressed<D> {
    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.0.k()
    }

    fn n(&self) -> Degree {
        self.0.n()
    }
    fn size(&self) -> usize {
        self.0.size()
    }
}

impl<D: Data> GLWEInfos for GLWEToLWESwitchingKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GLWEToLWESwitchingKeyCompressed<D> {
    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

impl<D: DataRef> fmt::Debug for GLWEToLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GLWEToLWESwitchingKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for GLWEToLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(GLWEToLWESwitchingKeyCompressed) {}", self.0)
    }
}

impl<D: DataMut> ReaderFrom for GLWEToLWESwitchingKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWEToLWESwitchingKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl GLWEToLWESwitchingKeyCompressed<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is unsupported for GLWEToLWESwitchingKeyCompressed"
        );
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is unsupported for GLWEToLWESwitchingKeyCompressed"
        );
        Self(GGLWESwitchingKeyCompressed::alloc(infos))
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum) -> Self {
        Self(GGLWESwitchingKeyCompressed::alloc_with(
            n,
            base2k,
            k,
            rank_in,
            Rank(1),
            dnum,
            Dsize(1),
        ))
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is unsupported for GLWEToLWESwitchingKeyCompressed"
        );
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is unsupported for GLWEToLWESwitchingKeyCompressed"
        );
        GGLWESwitchingKeyCompressed::alloc_bytes(infos)
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision, dnum: Dnum, rank_in: Rank) -> usize {
        GGLWESwitchingKeyCompressed::alloc_bytes_with(n, base2k, k, rank_in, dnum, Dsize(1))
    }
}
