use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    AutomorphismKey, Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    compressed::{Decompress, GLWESwitchingKeyCompressed, GLWESwitchingKeyCompressedToMut, GLWESwitchingKeyCompressedToRef},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct AutomorphismKeyCompressed<D: Data> {
    pub(crate) key: GLWESwitchingKeyCompressed<D>,
    pub(crate) p: i64,
}

impl<D: Data> LWEInfos for AutomorphismKeyCompressed<D> {
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
impl<D: Data> GLWEInfos for AutomorphismKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for AutomorphismKeyCompressed<D> {
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

impl<D: DataRef> fmt::Debug for AutomorphismKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for AutomorphismKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for AutomorphismKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(AutomorphismKeyCompressed: p={}) {}", self.p, self.key)
    }
}

impl AutomorphismKeyCompressed<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(infos.rank_in(), infos.rank_out());
        Self {
            key: GLWESwitchingKeyCompressed::alloc(infos),
            p: 0,
        }
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        Self {
            key: GLWESwitchingKeyCompressed::alloc_with(n, base2k, k, rank, rank, dnum, dsize),
            p: 0,
        }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(infos.rank_in(), infos.rank_out());
        GLWESwitchingKeyCompressed::alloc_bytes(infos)
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        GLWESwitchingKeyCompressed::alloc_bytes_with(n, base2k, k, rank, dnum, dsize)
    }
}

impl<D: DataMut> ReaderFrom for AutomorphismKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.p = reader.read_u64::<LittleEndian>()? as i64;
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for AutomorphismKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.p as u64)?;
        self.key.write_to(writer)
    }
}

impl<D: DataMut, DR: DataRef, B: Backend> Decompress<B, AutomorphismKeyCompressed<DR>> for AutomorphismKey<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &AutomorphismKeyCompressed<DR>) {
        self.key.decompress(module, &other.key);
        self.p = other.p;
    }
}

pub trait AutomorphismKeyCompressedToRef {
    fn to_ref(&self) -> AutomorphismKeyCompressed<&[u8]>;
}

impl<D: DataRef> AutomorphismKeyCompressedToRef for AutomorphismKeyCompressed<D>
where
    GLWESwitchingKeyCompressed<D>: GLWESwitchingKeyCompressedToRef,
{
    fn to_ref(&self) -> AutomorphismKeyCompressed<&[u8]> {
        AutomorphismKeyCompressed {
            key: self.key.to_ref(),
            p: self.p,
        }
    }
}

pub trait AutomorphismKeyCompressedToMut {
    fn to_mut(&mut self) -> AutomorphismKeyCompressed<&mut [u8]>;
}

impl<D: DataMut> AutomorphismKeyCompressedToMut for AutomorphismKeyCompressed<D>
where
    GLWESwitchingKeyCompressed<D>: GLWESwitchingKeyCompressedToMut,
{
    fn to_mut(&mut self) -> AutomorphismKeyCompressed<&mut [u8]> {
        AutomorphismKeyCompressed {
            p: self.p,
            key: self.key.to_mut(),
        }
    }
}
