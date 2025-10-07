use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dsize, GGLWEAutomorphismKey, GGLWEInfos, GLWEInfos, LWEInfos, Rank, Dnum, TorusPrecision,
    compressed::{Decompress, GGLWESwitchingKeyCompressed},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWEAutomorphismKeyCompressed<D: Data> {
    pub(crate) key: GGLWESwitchingKeyCompressed<D>,
    pub(crate) p: i64,
}

impl<D: Data> LWEInfos for GGLWEAutomorphismKeyCompressed<D> {
    fn n(&self) -> Degree {
        self.key.n()
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.key.k()
    }

    fn size(&self) -> usize {
        self.key.size()
    }
}
impl<D: Data> GLWEInfos for GGLWEAutomorphismKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GGLWEAutomorphismKeyCompressed<D> {
    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.key.dsize()
    }

    fn dnum(&self) -> Dnum {
        self.key.dnum()
    }
}

impl<D: DataRef> fmt::Debug for GGLWEAutomorphismKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GGLWEAutomorphismKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for GGLWEAutomorphismKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(AutomorphismKeyCompressed: p={}) {}", self.p, self.key)
    }
}

impl GGLWEAutomorphismKeyCompressed<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(infos.rank_in(), infos.rank_out());
        Self {
            key: GGLWESwitchingKeyCompressed::alloc(infos),
            p: 0,
        }
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        Self {
            key: GGLWESwitchingKeyCompressed::alloc_with(n, base2k, k, rank, rank, dnum, dsize),
            p: 0,
        }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(infos.rank_in(), infos.rank_out());
        GGLWESwitchingKeyCompressed::alloc_bytes(infos)
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        GGLWESwitchingKeyCompressed::alloc_bytes_with(n, base2k, k, rank, dnum, dsize)
    }
}

impl<D: DataMut> ReaderFrom for GGLWEAutomorphismKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.p = reader.read_u64::<LittleEndian>()? as i64;
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGLWEAutomorphismKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.p as u64)?;
        self.key.write_to(writer)
    }
}

impl<D: DataMut, DR: DataRef, B: Backend> Decompress<B, GGLWEAutomorphismKeyCompressed<DR>> for GGLWEAutomorphismKey<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &GGLWEAutomorphismKeyCompressed<DR>) {
        self.key.decompress(module, &other.key);
        self.p = other.p;
    }
}
