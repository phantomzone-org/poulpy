use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, WriterTo},
    source::Source,
};

use std::{fmt, marker::PhantomData};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_core::{
    Distribution,
    layouts::{Base2K, Degree, Digits, GGSWInfos, GLWEInfos, LWEInfos, TorusPrecision, compressed::GGSWCiphertextCompressed},
};

use crate::tfhe::blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos};

#[derive(Clone)]
pub struct BlindRotationKeyCompressed<D: Data, BRT: BlindRotationAlgo> {
    pub(crate) keys: Vec<GGSWCiphertextCompressed<D>>,
    pub(crate) dist: Distribution,
    pub(crate) _phantom: PhantomData<BRT>,
}

impl<D: DataRef, BRT: BlindRotationAlgo> fmt::Debug for BlindRotationKeyCompressed<D, BRT> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: Data, BRT: BlindRotationAlgo> PartialEq for BlindRotationKeyCompressed<D, BRT> {
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

impl<D: Data, BRT: BlindRotationAlgo> Eq for BlindRotationKeyCompressed<D, BRT> {}

impl<D: DataRef, BRT: BlindRotationAlgo> fmt::Display for BlindRotationKeyCompressed<D, BRT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "key[{i}]: {key}")?;
        }
        writeln!(f, "{:?}", self.dist)
    }
}

impl<D: DataMut, BRT: BlindRotationAlgo> FillUniform for BlindRotationKeyCompressed<D, BRT> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys
            .iter_mut()
            .for_each(|key| key.fill_uniform(log_bound, source));
    }
}

impl<D: DataMut, BRT: BlindRotationAlgo> ReaderFrom for BlindRotationKeyCompressed<D, BRT> {
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

impl<D: DataRef, BRT: BlindRotationAlgo> WriterTo for BlindRotationKeyCompressed<D, BRT> {
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

impl<D: DataRef, BRA: BlindRotationAlgo> BlindRotationKeyInfos for BlindRotationKeyCompressed<D, BRA> {
    fn n_glwe(&self) -> Degree {
        self.n()
    }

    fn n_lwe(&self) -> Degree {
        Degree(self.keys.len() as u32)
    }
}

impl<D: DataRef, BRA: BlindRotationAlgo> LWEInfos for BlindRotationKeyCompressed<D, BRA> {
    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn size(&self) -> usize {
        self.keys[0].size()
    }

    fn k(&self) -> TorusPrecision {
        self.keys[0].k()
    }

    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }
}

impl<D: DataRef, BRA: BlindRotationAlgo> GLWEInfos for BlindRotationKeyCompressed<D, BRA> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.keys[0].rank()
    }
}

impl<D: DataRef, BRA: BlindRotationAlgo> GGSWInfos for BlindRotationKeyCompressed<D, BRA> {
    fn rows(&self) -> poulpy_core::layouts::Rows {
        self.keys[0].rows()
    }

    fn digits(&self) -> poulpy_core::layouts::Digits {
        Digits(1)
    }
}

impl<D: DataRef, BRA: BlindRotationAlgo> BlindRotationKeyCompressed<D, BRA> {
    #[allow(dead_code)]
    pub(crate) fn block_size(&self) -> usize {
        match self.dist {
            Distribution::BinaryBlock(value) => value,
            _ => 1,
        }
    }
}
