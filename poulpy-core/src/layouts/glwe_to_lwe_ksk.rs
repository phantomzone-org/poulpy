use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{Base2K, Degree, Digits, GGLWEInfos, GGLWESwitchingKey, GLWEInfos, LWEInfos, Rank, Rows, TorusPrecision};

use std::fmt;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWEToLWEKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rows: Rows,
    pub rank_in: Rank,
}

impl LWEInfos for GLWEToLWEKeyLayout {
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

impl GLWEInfos for GLWEToLWEKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for GLWEToLWEKeyLayout {
    fn rank_in(&self) -> Rank {
        self.rank_in
    }

    fn digits(&self) -> Digits {
        Digits(1)
    }

    fn rank_out(&self) -> Rank {
        Rank(1)
    }

    fn rows(&self) -> Rows {
        self.rows
    }
}

/// A special [GLWESwitchingKey] required to for the conversion from [GLWECiphertext] to [LWECiphertext].
#[derive(PartialEq, Eq, Clone)]
pub struct GLWEToLWEKey<D: Data>(pub(crate) GGLWESwitchingKey<D>);

impl<D: Data> LWEInfos for GLWEToLWEKey<D> {
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

impl<D: Data> GLWEInfos for GLWEToLWEKey<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}
impl<D: Data> GGLWEInfos for GLWEToLWEKey<D> {
    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn digits(&self) -> Digits {
        self.0.digits()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn rows(&self) -> Rows {
        self.0.rows()
    }
}

impl<D: DataRef> fmt::Debug for GLWEToLWEKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GLWEToLWEKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for GLWEToLWEKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(GLWEToLWESwitchingKey) {}", self.0)
    }
}

impl<D: DataMut> ReaderFrom for GLWEToLWEKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWEToLWEKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl GLWEToLWEKey<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for GLWEToLWESwitchingKey"
        );
        debug_assert_eq!(
            infos.digits().0,
            1,
            "digits > 1 is not supported for GLWEToLWESwitchingKey"
        );
        Self(GGLWESwitchingKey::alloc(infos))
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision, rows: Rows, rank_in: Rank) -> Self {
        Self(GGLWESwitchingKey::alloc_with(
            n,
            base2k,
            k,
            rows,
            Digits(1),
            rank_in,
            Rank(1),
        ))
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for GLWEToLWESwitchingKey"
        );
        debug_assert_eq!(
            infos.digits().0,
            1,
            "digits > 1 is not supported for GLWEToLWESwitchingKey"
        );
        GGLWESwitchingKey::alloc_bytes(infos)
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision, rows: Rows, rank_in: Rank) -> usize {
        GGLWESwitchingKey::alloc_bytes_with(n, base2k, k, rows, Digits(1), rank_in, Rank(1))
    }
}
