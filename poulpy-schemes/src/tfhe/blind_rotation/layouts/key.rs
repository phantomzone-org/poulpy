use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, WriterTo},
    source::Source,
};

use std::{fmt, marker::PhantomData};

use poulpy_core::{
    Distribution,
    layouts::{Base2K, Degree, Dnum, Dsize, GGSW, GGSWInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::tfhe::blind_rotation::BlindRotationAlgo;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct BlindRotationKeyLayout {
    pub n_glwe: Degree,
    pub n_lwe: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub dnum: Dnum,
    pub rank: Rank,
}

impl BlindRotationKeyInfos for BlindRotationKeyLayout {
    fn n_glwe(&self) -> Degree {
        self.n_glwe
    }

    fn n_lwe(&self) -> Degree {
        self.n_lwe
    }
}

impl GGSWInfos for BlindRotationKeyLayout {
    fn dsize(&self) -> Dsize {
        Dsize(1)
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

impl GLWEInfos for BlindRotationKeyLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl LWEInfos for BlindRotationKeyLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        self.n_glwe
    }
}

pub trait BlindRotationKeyInfos
where
    Self: GGSWInfos,
{
    fn n_glwe(&self) -> Degree;
    fn n_lwe(&self) -> Degree;
}

pub trait BlindRotationKeyAlloc {
    fn alloc<A>(infos: &A) -> Self
    where
        A: BlindRotationKeyInfos;
}

#[derive(Clone)]
pub struct BlindRotationKey<D: Data, BRT: BlindRotationAlgo> {
    pub(crate) keys: Vec<GGSW<D>>,
    pub(crate) dist: Distribution,
    pub(crate) _phantom: PhantomData<BRT>,
}

pub trait BlindRotationKeyFactory<BRA: BlindRotationAlgo> {
    fn blind_rotation_key_alloc<A>(infos: &A) -> BlindRotationKey<Vec<u8>, BRA>
    where
        A: BlindRotationKeyInfos;
}

impl<BRA: BlindRotationAlgo> BlindRotationKey<Vec<u8>, BRA>
where
    Self: BlindRotationKeyFactory<BRA>,
{
    pub fn alloc<A>(infos: &A) -> BlindRotationKey<Vec<u8>, BRA>
    where
        A: BlindRotationKeyInfos,
    {
        Self::blind_rotation_key_alloc(infos)
    }
}

impl<D: DataRef, BRT: BlindRotationAlgo> fmt::Debug for BlindRotationKey<D, BRT> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: Data, BRT: BlindRotationAlgo> PartialEq for BlindRotationKey<D, BRT> {
    fn eq(&self, other: &Self) -> bool {
        if self.keys.len() != other.keys.len() {
            return false;
        }
        for (a, b) in self.keys.iter().zip(other.keys.iter()) {
            if a != b {
                return false;
            }
        }

        self.dist == other.dist && self._phantom == other._phantom
    }
}

impl<D: Data, BRT: BlindRotationAlgo> Eq for BlindRotationKey<D, BRT> {}

impl<D: DataRef, BRT: BlindRotationAlgo> fmt::Display for BlindRotationKey<D, BRT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "key[{i}]: {key}")?;
        }
        writeln!(f, "{:?}", self.dist)
    }
}

impl<D: DataMut, BRT: BlindRotationAlgo> FillUniform for BlindRotationKey<D, BRT> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys
            .iter_mut()
            .for_each(|key| key.fill_uniform(log_bound, source));
    }
}

impl<D: DataMut, BRT: BlindRotationAlgo> ReaderFrom for BlindRotationKey<D, BRT> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        match Distribution::read_from(reader) {
            Ok(dist) => self.dist = dist,
            Err(e) => return Err(e),
        }
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

impl<D: DataRef, BRT: BlindRotationAlgo> WriterTo for BlindRotationKey<D, BRT> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        match self.dist.write_to(writer) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

impl<D: DataRef, BRT: BlindRotationAlgo> BlindRotationKeyInfos for BlindRotationKey<D, BRT> {
    fn n_glwe(&self) -> Degree {
        self.n()
    }

    fn n_lwe(&self) -> Degree {
        Degree(self.keys.len() as u32)
    }
}

impl<D: DataRef, BRT: BlindRotationAlgo> BlindRotationKey<D, BRT> {
    #[allow(dead_code)]
    fn block_size(&self) -> usize {
        match self.dist {
            Distribution::BinaryBlock(value) => value,
            _ => 1,
        }
    }
}

impl<D: DataRef, BRT: BlindRotationAlgo> LWEInfos for BlindRotationKey<D, BRT> {
    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.keys[0].k()
    }

    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn size(&self) -> usize {
        self.keys[0].size()
    }
}

impl<D: DataRef, BRT: BlindRotationAlgo> GLWEInfos for BlindRotationKey<D, BRT> {
    fn rank(&self) -> Rank {
        self.keys[0].rank()
    }
}
impl<D: DataRef, BRT: BlindRotationAlgo> GGSWInfos for BlindRotationKey<D, BRT> {
    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        Dsize(1)
    }

    fn dnum(&self) -> Dnum {
        self.keys[0].dnum()
    }
}
