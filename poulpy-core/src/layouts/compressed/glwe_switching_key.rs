use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWECompressedSeedMut, GGLWEInfos, GGLWEToMut, GLWEInfos, GLWESwitchingKey,
    GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, LWEInfos, Rank, TorusPrecision,
    compressed::{GGLWECompressed, GGLWECompressedToMut, GGLWECompressedToRef, GGLWEDecompress},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GLWESwitchingKeyCompressed<D: Data> {
    pub(crate) key: GGLWECompressed<D>,
    pub(crate) input_degree: Degree,  // Degree of sk_in
    pub(crate) output_degree: Degree, // Degree of sk_out
}

impl<D: DataMut> GGLWECompressedSeedMut for GLWESwitchingKeyCompressed<D> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.key.seed
    }
}

impl<D: DataRef> GLWESwitchingKeyDegrees for GLWESwitchingKeyCompressed<D> {
    fn output_degree(&self) -> &Degree {
        &self.output_degree
    }

    fn input_degree(&self) -> &Degree {
        &self.input_degree
    }
}

impl<D: DataMut> GLWESwitchingKeyDegreesMut for GLWESwitchingKeyCompressed<D> {
    fn output_degree(&mut self) -> &mut Degree {
        &mut self.output_degree
    }

    fn input_degree(&mut self) -> &mut Degree {
        &mut self.input_degree
    }
}

impl<D: Data> LWEInfos for GLWESwitchingKeyCompressed<D> {
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
impl<D: Data> GLWEInfos for GLWESwitchingKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GLWESwitchingKeyCompressed<D> {
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

impl<D: DataRef> fmt::Debug for GLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GLWESwitchingKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for GLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GLWESwitchingKeyCompressed: sk_in_n={} sk_out_n={}) {}",
            self.input_degree, self.output_degree, self.key.data
        )
    }
}

impl GLWESwitchingKeyCompressed<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank_in: Rank, rank_out: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        GLWESwitchingKeyCompressed {
            key: GGLWECompressed::alloc(n, base2k, k, rank_in, rank_out, dnum, dsize),
            input_degree: Degree(0),
            output_degree: Degree(0),
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        GGLWECompressed::bytes_of_from_infos(infos)
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum, dsize: Dsize) -> usize
where {
        GGLWECompressed::bytes_of(n, base2k, k, rank_in, dnum, dsize)
    }
}

impl<D: DataMut> ReaderFrom for GLWESwitchingKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.input_degree = Degree(reader.read_u32::<LittleEndian>()?);
        self.output_degree = Degree(reader.read_u32::<LittleEndian>()?);
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWESwitchingKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.input_degree.into())?;
        writer.write_u32::<LittleEndian>(self.output_degree.into())?;
        self.key.write_to(writer)
    }
}

pub trait GLWESwitchingKeyDecompress
where
    Self: GGLWEDecompress,
{
    fn decompress_glwe_switching_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToRef + GLWESwitchingKeyDegrees,
    {
        self.decompress_gglwe(res, other);

        *res.input_degree() = *other.input_degree();
        *res.output_degree() = *other.output_degree();
    }
}

impl<B: Backend> GLWESwitchingKeyDecompress for Module<B> where Self: GGLWEDecompress {}

impl<D: DataMut> GLWESwitchingKey<D> {
    pub fn decompress<O, M>(&mut self, module: &M, other: &O)
    where
        O: GGLWECompressedToRef + GLWESwitchingKeyDegrees,
        M: GLWESwitchingKeyDecompress,
    {
        module.decompress_glwe_switching_key(self, other);
    }
}

impl<D: DataMut> GGLWECompressedToMut for GLWESwitchingKeyCompressed<D> {
    fn to_mut(&mut self) -> GGLWECompressed<&mut [u8]> {
        self.key.to_mut()
    }
}

impl<D: DataRef> GGLWECompressedToRef for GLWESwitchingKeyCompressed<D> {
    fn to_ref(&self) -> GGLWECompressed<&[u8]> {
        self.key.to_ref()
    }
}
