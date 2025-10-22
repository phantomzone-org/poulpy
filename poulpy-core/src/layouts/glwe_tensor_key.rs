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
pub struct GLWETensorKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

#[derive(PartialEq, Eq, Clone)]
pub struct GLWETensorKey<D: Data> {
    pub(crate) keys: Vec<GGLWE<D>>,
}

impl<D: Data> LWEInfos for GLWETensorKey<D> {
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

impl<D: Data> GLWEInfos for GLWETensorKey<D> {
    fn rank(&self) -> Rank {
        self.keys[0].rank_out()
    }
}

impl<D: Data> GGLWEInfos for GLWETensorKey<D> {
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

impl LWEInfos for GLWETensorKeyLayout {
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

impl GLWEInfos for GLWETensorKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for GLWETensorKeyLayout {
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

impl<D: DataRef> fmt::Debug for GLWETensorKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GLWETensorKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys
            .iter_mut()
            .for_each(|key: &mut GGLWE<D>| key.fill_uniform(log_bound, source))
    }
}

impl<D: DataRef> fmt::Display for GLWETensorKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GLWETensorKey)",)?;
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "{i}: {key}")?;
        }
        Ok(())
    }
}

impl GLWETensorKey<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKey"
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
        let pairs: u32 = (((rank.0 + 1) * rank.0) >> 1).max(1);
        GLWETensorKey {
            keys: (0..pairs)
                .map(|_| GGLWE::alloc(n, base2k, k, Rank(1), rank, dnum, dsize))
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
            "rank_in != rank_out is not supported for GGLWETensorKey"
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
        let pairs: usize = (((rank.0 + 1) * rank.0) >> 1).max(1) as usize;
        pairs * GGLWE::bytes_of(n, base2k, k, Rank(1), rank, dnum, dsize)
    }
}

impl<D: DataMut> GLWETensorKey<D> {
    // Returns a mutable reference to GGLWE_{s}(s[i] * s[j])
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GGLWE<D> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank_out().into();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataRef> GLWETensorKey<D> {
    // Returns a reference to GGLWE_{s}(s[i] * s[j])
    pub fn at(&self, mut i: usize, mut j: usize) -> &GGLWE<D> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank_out().into();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataMut> ReaderFrom for GLWETensorKey<D> {
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

impl<D: DataRef> WriterTo for GLWETensorKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

pub trait GLWETensorKeyToRef {
    fn to_ref(&self) -> GLWETensorKey<&[u8]>;
}

impl<D: DataRef> GLWETensorKeyToRef for GLWETensorKey<D>
where
    GGLWE<D>: GGLWEToRef,
{
    fn to_ref(&self) -> GLWETensorKey<&[u8]> {
        GLWETensorKey {
            keys: self.keys.iter().map(|c| c.to_ref()).collect(),
        }
    }
}

pub trait GLWETensorKeyToMut {
    fn to_mut(&mut self) -> GLWETensorKey<&mut [u8]>;
}

impl<D: DataMut> GLWETensorKeyToMut for GLWETensorKey<D>
where
    GGLWE<D>: GGLWEToMut,
{
    fn to_mut(&mut self) -> GLWETensorKey<&mut [u8]> {
        GLWETensorKey {
            keys: self.keys.iter_mut().map(|c| c.to_mut()).collect(),
        }
    }
}
