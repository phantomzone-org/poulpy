use std::fmt;

use backend::hal::{
    api::{FillUniform, Reset, ZnxInfos},
    layouts::{Data, DataMut, DataRef, ReaderFrom, VecZnx, VecZnxToMut, VecZnxToRef, WriterTo},
};
use sampling::source::Source;

use crate::{Infos, SetMetaData};

#[derive(PartialEq, Eq, Clone)]
pub struct LWECiphertext<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) k: usize,
    pub(crate) basek: usize,
}

impl<D: DataRef> fmt::Debug for LWECiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
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

impl<D: DataMut> Reset for LWECiphertext<D>
where
    VecZnx<D>: Reset,
{
    fn reset(&mut self) {
        self.data.reset();
        self.basek = 0;
        self.k = 0;
    }
}

impl<D: DataMut> FillUniform for LWECiphertext<D>
where
    VecZnx<D>: FillUniform,
{
    fn fill_uniform(&mut self, source: &mut Source) {
        self.data.fill_uniform(source);
    }
}

impl LWECiphertext<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize) -> Self {
        Self {
            data: VecZnx::alloc(n + 1, 1, k.div_ceil(basek)),
            k: k,
            basek: basek,
        }
    }
}

impl<D: Data> Infos for LWECiphertext<D>
where
    VecZnx<D>: ZnxInfos,
{
    type Inner = VecZnx<D>;

    fn n(&self) -> usize {
        &self.inner().n() - 1
    }

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<DataSelf: DataMut> SetMetaData for LWECiphertext<DataSelf> {
    fn set_k(&mut self, k: usize) {
        self.k = k
    }

    fn set_basek(&mut self, basek: usize) {
        self.basek = basek
    }
}

pub trait LWECiphertextToRef {
    fn to_ref(&self) -> LWECiphertext<&[u8]>;
}

impl<D: DataRef> LWECiphertextToRef for LWECiphertext<D> {
    fn to_ref(&self) -> LWECiphertext<&[u8]> {
        LWECiphertext {
            data: self.data.to_ref(),
            basek: self.basek,
            k: self.k,
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
            basek: self.basek,
            k: self.k,
        }
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut> ReaderFrom for LWECiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWECiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        self.data.write_to(writer)
    }
}
