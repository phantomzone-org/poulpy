use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, LWEToGLWESwitchingKey, Rank, Dnum, TorusPrecision,
    compressed::{Decompress, GGLWESwitchingKeyCompressed},
};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct LWEToGLWESwitchingKeyCompressed<D: Data>(pub(crate) GGLWESwitchingKeyCompressed<D>);

impl<D: Data> LWEInfos for LWEToGLWESwitchingKeyCompressed<D> {
    fn n(&self) -> Degree {
        self.0.n()
    }

    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.0.k()
    }
    fn size(&self) -> usize {
        self.0.size()
    }
}
impl<D: Data> GLWEInfos for LWEToGLWESwitchingKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for LWEToGLWESwitchingKeyCompressed<D> {
    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

impl<D: DataRef> fmt::Debug for LWEToGLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for LWEToGLWESwitchingKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for LWEToGLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(LWEToGLWESwitchingKeyCompressed) {}", self.0)
    }
}

impl<D: DataMut> ReaderFrom for LWEToGLWESwitchingKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWEToGLWESwitchingKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl LWEToGLWESwitchingKeyCompressed<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWEToGLWESwitchingKeyCompressed"
        );
        debug_assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWEToGLWESwitchingKeyCompressed"
        );
        Self(GGLWESwitchingKeyCompressed::alloc(infos))
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> Self {
        Self(GGLWESwitchingKeyCompressed::alloc_with(
            n,
            base2k,
            k,
            Rank(1),
            rank_out,
            dnum,
            Dsize(1),
        ))
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWEToGLWESwitchingKey"
        );
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWEToGLWESwitchingKey"
        );
        GGLWESwitchingKeyCompressed::alloc_bytes(infos)
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> usize {
        GGLWESwitchingKeyCompressed::alloc_bytes_with(n, base2k, k, Rank(1), dnum, Dsize(1))
    }
}

impl<D: DataMut, DR: DataRef, B: Backend> Decompress<B, LWEToGLWESwitchingKeyCompressed<DR>> for LWEToGLWESwitchingKey<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &LWEToGLWESwitchingKeyCompressed<DR>) {
        self.0.decompress(module, &other.0);
    }
}
