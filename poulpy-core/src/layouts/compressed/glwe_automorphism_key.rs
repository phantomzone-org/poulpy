use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWECompressed, GGLWECompressedSeedMut, GGLWECompressedToBackendMut,
    GGLWECompressedToBackendRef, GGLWECompressedToMut, GGLWECompressedToRef, GGLWEDecompress, GGLWEInfos, GLWEDecompress,
    GLWEInfos, GetGaloisElement, LWEInfos, Rank, SetGaloisElement, TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

/// Seed-compressed GLWE automorphism key layout.
///
/// Wraps a [`GGLWECompressed`] with a Galois element `p` for applying
/// automorphisms `X → X^p` on GLWE ciphertexts.
#[derive(PartialEq, Eq, Clone)]
pub struct GLWEAutomorphismKeyCompressed<D: Data> {
    pub(crate) key: GGLWECompressed<D>,
    pub(crate) p: i64,
}

impl<D: HostDataRef> GetGaloisElement for GLWEAutomorphismKeyCompressed<D> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data> LWEInfos for GLWEAutomorphismKeyCompressed<D> {
    fn n(&self) -> Degree {
        self.key.n()
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn size(&self) -> usize {
        self.key.size()
    }
}

impl<D: Data> GLWEInfos for GLWEAutomorphismKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GLWEAutomorphismKeyCompressed<D> {
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

impl<D: HostDataRef> fmt::Debug for GLWEAutomorphismKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut> FillUniform for GLWEAutomorphismKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef> fmt::Display for GLWEAutomorphismKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(AutomorphismKeyCompressed: p={}) {}", self.p, self.key)
    }
}

impl GLWEAutomorphismKeyCompressed<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        GLWEAutomorphismKeyCompressed {
            key: GGLWECompressed::alloc(n, base2k, k, rank, rank, dnum, dsize),
            p: 0,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        GGLWECompressed::bytes_of(n, base2k, k, rank, dnum, dsize)
    }
}

impl<D: HostDataMut> ReaderFrom for GLWEAutomorphismKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.p = reader.read_u64::<LittleEndian>()? as i64;
        self.key.read_from(reader)
    }
}

impl<D: HostDataRef> WriterTo for GLWEAutomorphismKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.p as u64)?;
        self.key.write_to(writer)
    }
}

pub trait GLWEAutomorphismKeyDecompress
where
    Self: GGLWEDecompress,
{
    fn decompress_automorphism_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: crate::layouts::GGLWEToBackendMut<Self::Backend> + SetGaloisElement,
        O: GGLWECompressedToBackendRef<Self::Backend> + GetGaloisElement,
    {
        self.decompress_gglwe(res, other);
        res.set_p(other.p());
    }
}

impl<B: Backend> GLWEAutomorphismKeyDecompress for Module<B> where Self: GLWEDecompress {}

// module-only API: decompression is provided by `GLWEAutomorphismKeyDecompress` on `Module`.

impl<D: HostDataRef> GGLWECompressedToRef for GLWEAutomorphismKeyCompressed<D> {
    fn to_ref(&self) -> GGLWECompressed<&[u8]> {
        self.key.to_ref()
    }
}

impl<D: HostDataMut> GGLWECompressedToMut for GLWEAutomorphismKeyCompressed<D> {
    fn to_mut(&mut self) -> GGLWECompressed<&mut [u8]> {
        self.key.to_mut()
    }
}

impl<BE: Backend> GGLWECompressedToBackendRef<BE> for GLWEAutomorphismKeyCompressed<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> crate::layouts::compressed::GGLWECompressedBackendRef<'_, BE> {
        <crate::layouts::GGLWECompressed<BE::OwnedBuf> as GGLWECompressedToBackendRef<BE>>::to_backend_ref(&self.key)
    }
}

impl<BE: Backend> GGLWECompressedToBackendMut<BE> for GLWEAutomorphismKeyCompressed<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> crate::layouts::compressed::GGLWECompressedBackendMut<'_, BE> {
        <crate::layouts::GGLWECompressed<BE::OwnedBuf> as GGLWECompressedToBackendMut<BE>>::to_backend_mut(&mut self.key)
    }
}

impl<D: HostDataMut> GGLWECompressedSeedMut for GLWEAutomorphismKeyCompressed<D> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.key.seed
    }
}

impl<D: HostDataMut> SetGaloisElement for GLWEAutomorphismKeyCompressed<D> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}
