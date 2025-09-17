use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, Reset, Scratch, WriterTo},
    source::Source,
};

use std::{fmt, marker::PhantomData};

use poulpy_core::{
    Distribution,
    layouts::{GGSWCiphertext, Infos, LWESecret, prepared::GLWESecretPrepared},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::tfhe::blind_rotation::BlindRotationAlgo;

pub trait BlindRotationKeyAlloc {
    fn alloc(n_gglwe: usize, n_lwe: usize, basek: usize, k: usize, rows: usize, rank: usize) -> Self;
}

pub trait BlindRotationKeyEncryptSk<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn encrypt_sk<DataSkGLWE, DataSkLWE>(
        &mut self,
        module: &Module<B>,
        sk_glwe: &GLWESecretPrepared<DataSkGLWE, B>,
        sk_lwe: &LWESecret<DataSkLWE>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        DataSkGLWE: DataRef,
        DataSkLWE: DataRef;
}

#[derive(Clone)]
pub struct BlindRotationKey<D: Data, BRT: BlindRotationAlgo> {
    pub(crate) keys: Vec<GGSWCiphertext<D>>,
    pub(crate) dist: Distribution,
    pub(crate) _phantom: PhantomData<BRT>,
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

impl<D: DataMut, BRT: BlindRotationAlgo> Reset for BlindRotationKey<D, BRT> {
    fn reset(&mut self) {
        self.keys.iter_mut().for_each(|key| key.reset());
        self.dist = Distribution::NONE;
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

impl<D: DataRef, BRT: BlindRotationAlgo> BlindRotationKey<D, BRT> {
    #[allow(dead_code)]
    pub(crate) fn n(&self) -> usize {
        self.keys[0].n()
    }

    #[allow(dead_code)]
    pub(crate) fn rows(&self) -> usize {
        self.keys[0].rows()
    }

    #[allow(dead_code)]
    pub(crate) fn k(&self) -> usize {
        self.keys[0].k()
    }

    #[allow(dead_code)]
    pub(crate) fn size(&self) -> usize {
        self.keys[0].size()
    }

    #[allow(dead_code)]
    pub(crate) fn rank(&self) -> usize {
        self.keys[0].rank()
    }

    pub(crate) fn basek(&self) -> usize {
        self.keys[0].basek()
    }

    #[allow(dead_code)]
    pub(crate) fn block_size(&self) -> usize {
        match self.dist {
            Distribution::BinaryBlock(value) => value,
            _ => 1,
        }
    }
}
