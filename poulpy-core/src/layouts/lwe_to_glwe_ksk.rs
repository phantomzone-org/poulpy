use std::fmt;

use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{Base2K, Degree, Digits, GGLWEInfos, GGLWESwitchingKey, GLWEInfos, LWEInfos, Rank, Rows, TorusPrecision};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct LWEToGLWESwitchingKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rows: Rows,
    pub rank_out: Rank,
}

impl LWEInfos for LWEToGLWESwitchingKeyLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        self.n
    }
}

impl GLWEInfos for LWEToGLWESwitchingKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for LWEToGLWESwitchingKeyLayout {
    fn rank_in(&self) -> Rank {
        Rank(1)
    }

    fn digits(&self) -> Digits {
        Digits(1)
    }

    fn rank_out(&self) -> Rank {
        self.rank_out
    }

    fn rows(&self) -> Rows {
        self.rows
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct LWEToGLWESwitchingKey<D: Data>(pub(crate) GGLWESwitchingKey<D>);

impl<D: Data> LWEInfos for LWEToGLWESwitchingKey<D> {
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

impl<D: Data> GLWEInfos for LWEToGLWESwitchingKey<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}
impl<D: Data> GGLWEInfos for LWEToGLWESwitchingKey<D> {
    fn digits(&self) -> Digits {
        self.0.digits()
    }

    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn rows(&self) -> Rows {
        self.0.rows()
    }
}

impl<D: DataRef> fmt::Debug for LWEToGLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for LWEToGLWESwitchingKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for LWEToGLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(LWEToGLWESwitchingKey) {}", self.0)
    }
}

impl<D: DataMut> ReaderFrom for LWEToGLWESwitchingKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWEToGLWESwitchingKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl LWEToGLWESwitchingKey<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWEToGLWESwitchingKey"
        );
        debug_assert_eq!(
            infos.digits().0,
            1,
            "digits > 1 is not supported for LWEToGLWESwitchingKey"
        );
        Self(GGLWESwitchingKey::alloc(infos))
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision, rows: Rows, rank_out: Rank) -> Self {
        Self(GGLWESwitchingKey::alloc_with(
            n,
            base2k,
            k,
            rows,
            Digits(1),
            Rank(1),
            rank_out,
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
            infos.digits().0,
            1,
            "digits > 1 is not supported for LWEToGLWESwitchingKey"
        );
        GGLWESwitchingKey::alloc_bytes(infos)
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision, rows: Rows, rank_out: Rank) -> usize {
        GGLWESwitchingKey::alloc_bytes_with(n, base2k, k, rows, Digits(1), Rank(1), rank_out)
    }
}
