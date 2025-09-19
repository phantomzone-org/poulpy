use std::fmt;

use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, Reset, WriterTo, Zn, ZnToMut, ZnToRef, ZnxInfos},
    source::Source,
};

#[derive(PartialEq, Eq, Clone)]
pub struct LWECiphertext<D: Data> {
    pub(crate) data: Zn<D>,
    pub(crate) metadata: LWEMetadata,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct LWEMetadata {
    pub k: usize,
    pub basek: usize,
}

impl LWEMetadata {
    pub fn as_glwe(&self) -> GLWEMetadata {
        GLWEMetadata {
            basek: self.basek,
            k: self.k,
            rank: 1,
        }
    }
}

impl<D: DataRef> LWECiphertext<D> {
    pub fn data(&self) -> &Zn<D> {
        &self.data
    }
}

impl<D: DataMut> LWECiphertext<D> {
    pub fn data_mut(&mut self) -> &Zn<D> {
        &mut self.data
    }
}

impl<D: DataRef> fmt::Debug for LWECiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef> fmt::Display for LWECiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LWECiphertext: basek={} k={}: {}",
            self.basek(),
            self.k(),
            self.data
        )
    }
}

impl<D: DataMut> Reset for LWECiphertext<D> {
    fn reset(&mut self) {
        self.data.reset();
        self.metadata.basek = 0;
        self.metadata.k = 0;
    }
}

impl<D: DataMut> FillUniform for LWECiphertext<D>
where
    Zn<D>: FillUniform,
{
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl LWECiphertext<Vec<u8>> {
    pub fn alloc(n: usize, metadata: LWEMetadata) -> Self {
        Self {
            data: Zn::alloc(n + 1, 1, metadata.k.div_ceil(metadata.basek)),
            metadata,
        }
    }
}

impl<D: Data> Infos for LWECiphertext<D>
where
    Zn<D>: ZnxInfos,
{
    type Inner = Zn<D>;

    fn n(&self) -> usize {
        &self.inner().n() - 1
    }

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.metadata.basek
    }

    fn k(&self) -> usize {
        self.metadata.k
    }
}

impl<DataSelf: DataMut> SetMetaData for LWECiphertext<DataSelf> {
    fn set_k(&mut self, k: usize) {
        self.metadata.k = k
    }

    fn set_basek(&mut self, basek: usize) {
        self.metadata.basek = basek
    }
}

pub trait LWECiphertextToRef {
    fn to_ref(&self) -> LWECiphertext<&[u8]>;
}

impl<D: DataRef> LWECiphertextToRef for LWECiphertext<D> {
    fn to_ref(&self) -> LWECiphertext<&[u8]> {
        LWECiphertext {
            data: self.data.to_ref(),
            metadata: self.metadata,
        }
    }
}

pub trait LWECiphertextToMut {
    #[allow(dead_code)]
    fn to_mut(&mut self) -> LWECiphertext<&mut [u8]>;
}

impl<D: DataMut> LWECiphertextToMut for LWECiphertext<D> {
    fn to_mut(&mut self) -> LWECiphertext<&mut [u8]> {
        LWECiphertext {
            data: self.data.to_mut(),
            metadata: self.metadata,
        }
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::layouts::{GLWEMetadata, Infos, SetMetaData};

impl<D: DataMut> ReaderFrom for LWECiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.metadata.k = reader.read_u64::<LittleEndian>()? as usize;
        self.metadata.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWECiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.metadata.k as u64)?;
        writer.write_u64::<LittleEndian>(self.metadata.basek as u64)?;
        self.data.write_to(writer)
    }
}
