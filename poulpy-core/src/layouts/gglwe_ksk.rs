use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEInfos, GGLWEToMut, GGLWEToRef, GLWE, GLWEInfos, LWEInfos, Rank, TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWESwitchingKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank_in: Rank,
    pub rank_out: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

impl LWEInfos for GLWESwitchingKeyLayout {
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

impl GLWEInfos for GLWESwitchingKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for GLWESwitchingKeyLayout {
    fn rank_in(&self) -> Rank {
        self.rank_in
    }

    fn rank_out(&self) -> Rank {
        self.rank_out
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GLWESwitchingKey<D: Data> {
    pub(crate) key: GGLWE<D>,
    pub(crate) input_degree: Degree,  // Degree of sk_in
    pub(crate) output_degree: Degree, // Degree of sk_out
}

pub trait GLWESwitchingKeyDegrees {
    fn input_degree(&self) -> &Degree;
    fn output_degree(&self) -> &Degree;
}

impl<D: DataRef> GLWESwitchingKeyDegrees for GLWESwitchingKey<D> {
    fn output_degree(&self) -> &Degree {
        &self.output_degree
    }

    fn input_degree(&self) -> &Degree {
        &self.input_degree
    }
}

pub trait GLWESwitchingKeyDegreesMut {
    fn input_degree(&mut self) -> &mut Degree;
    fn output_degree(&mut self) -> &mut Degree;
}

impl<D: DataMut> GLWESwitchingKeyDegreesMut for GLWESwitchingKey<D> {
    fn output_degree(&mut self) -> &mut Degree {
        &mut self.output_degree
    }

    fn input_degree(&mut self) -> &mut Degree {
        &mut self.input_degree
    }
}

impl<D: Data> LWEInfos for GLWESwitchingKey<D> {
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

impl<D: Data> GLWEInfos for GLWESwitchingKey<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GLWESwitchingKey<D> {
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

impl<D: DataRef> fmt::Debug for GLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef> fmt::Display for GLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GLWESwitchingKey: sk_in_n={} sk_out_n={}) {}",
            self.input_degree,
            self.output_degree,
            self.key.data()
        )
    }
}

impl<D: DataMut> FillUniform for GLWESwitchingKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.key.fill_uniform(log_bound, source);
    }
}

impl GLWESwitchingKey<Vec<u8>> {
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
        GLWESwitchingKey {
            key: GGLWE::alloc(n, base2k, k, rank_in, rank_out, dnum, dsize),
            input_degree: Degree(0),
            output_degree: Degree(0),
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn bytes_of(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        GGLWE::bytes_of(n, base2k, k, rank_in, rank_out, dnum, dsize)
    }
}

impl<D: DataMut> GGLWEToMut for GLWESwitchingKey<D> {
    fn to_mut(&mut self) -> GGLWE<&mut [u8]> {
        self.key.to_mut()
    }
}

impl<D: DataRef> GGLWEToRef for GLWESwitchingKey<D> {
    fn to_ref(&self) -> GGLWE<&[u8]> {
        self.key.to_ref()
    }
}

impl<D: DataRef> GLWESwitchingKey<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8]> {
        self.key.at(row, col)
    }
}

impl<D: DataMut> GLWESwitchingKey<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8]> {
        self.key.at_mut(row, col)
    }
}

impl<D: DataMut> ReaderFrom for GLWESwitchingKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.input_degree = Degree(reader.read_u32::<LittleEndian>()? as u32);
        self.output_degree = Degree(reader.read_u32::<LittleEndian>()? as u32);
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWESwitchingKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.input_degree.into())?;
        writer.write_u32::<LittleEndian>(self.output_degree.into())?;
        self.key.write_to(writer)
    }
}
