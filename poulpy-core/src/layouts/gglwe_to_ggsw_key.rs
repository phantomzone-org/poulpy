use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEInfos, GGLWEToMut, GGLWEToRef, GLWEInfos, LWEInfos, Rank, TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGLWEToGGSWKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWEToGGSWKey<D: Data> {
    pub(crate) keys: Vec<GGLWE<D>>,
}

impl<D: Data> LWEInfos for GGLWEToGGSWKey<D> {
    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.keys[0].k()
    }

    fn size(&self) -> usize {
        self.keys[0].size()
    }
}

impl<D: Data> GLWEInfos for GGLWEToGGSWKey<D> {
    fn rank(&self) -> Rank {
        self.keys[0].rank_out()
    }
}

impl<D: Data> GGLWEInfos for GGLWEToGGSWKey<D> {
    fn rank_in(&self) -> Rank {
        self.rank_out()
    }

    fn rank_out(&self) -> Rank {
        self.keys[0].rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.keys[0].dsize()
    }

    fn dnum(&self) -> Dnum {
        self.keys[0].dnum()
    }
}

impl LWEInfos for GGLWEToGGSWKeyLayout {
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

impl GLWEInfos for GGLWEToGGSWKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for GGLWEToGGSWKeyLayout {
    fn rank_in(&self) -> Rank {
        self.rank
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

impl<D: DataRef> fmt::Debug for GGLWEToGGSWKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GGLWEToGGSWKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys
            .iter_mut()
            .for_each(|key: &mut GGLWE<D>| key.fill_uniform(log_bound, source))
    }
}

impl<D: DataRef> fmt::Display for GGLWEToGGSWKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GGLWEToGGSWKey)",)?;
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "{i}: {key}")?;
        }
        Ok(())
    }
}

impl GGLWEToGGSWKey<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKey"
        );
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
        GGLWEToGGSWKey {
            keys: (0..rank.as_usize())
                .map(|_| GGLWE::alloc(n, base2k, k, rank, rank, dnum, dsize))
                .collect(),
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKey"
        );
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
        rank.as_usize() * GGLWE::bytes_of(n, base2k, k, rank, rank, dnum, dsize)
    }
}

impl<D: DataMut> GGLWEToGGSWKey<D> {
    // Returns a mutable reference to GGLWE_{s}([s[i]*s[0], s[i]*s[1], ..., s[i]*s[rank]])
    pub fn at_mut(&mut self, i: usize) -> &mut GGLWE<D> {
        assert!((i as u32) < self.rank());
        &mut self.keys[i]
    }
}

impl<D: DataRef> GGLWEToGGSWKey<D> {
    // Returns a reference to GGLWE_{s}(s[i] * s[j])
    pub fn at(&self, i: usize) -> &GGLWE<D> {
        assert!((i as u32) < self.rank());
        &self.keys[i]
    }
}

impl<D: DataMut> ReaderFrom for GGLWEToGGSWKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let len: usize = reader.read_u64::<LittleEndian>()? as usize;
        if self.keys.len() != len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("self.keys.len()={} != read len={}", self.keys.len(), len),
            ));
        }
        for key in &mut self.keys {
            key.read_from(reader)?;
        }
        Ok(())
    }
}

impl<D: DataRef> WriterTo for GGLWEToGGSWKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

pub trait GGLWEToGGSWKeyToRef {
    fn to_ref(&self) -> GGLWEToGGSWKey<&[u8]>;
}

impl<D: DataRef> GGLWEToGGSWKeyToRef for GGLWEToGGSWKey<D>
where
    GGLWE<D>: GGLWEToRef,
{
    fn to_ref(&self) -> GGLWEToGGSWKey<&[u8]> {
        GGLWEToGGSWKey {
            keys: self.keys.iter().map(|c| c.to_ref()).collect(),
        }
    }
}

pub trait GGLWEToGGSWKeyToMut {
    fn to_mut(&mut self) -> GGLWEToGGSWKey<&mut [u8]>;
}

impl<D: DataMut> GGLWEToGGSWKeyToMut for GGLWEToGGSWKey<D>
where
    GGLWE<D>: GGLWEToMut,
{
    fn to_mut(&mut self) -> GGLWEToGGSWKey<&mut [u8]> {
        GGLWEToGGSWKey {
            keys: self.keys.iter_mut().map(|c| c.to_mut()).collect(),
        }
    }
}
