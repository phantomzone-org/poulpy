use poulpy_hal::{
    api::{FillUniform, Reset},
    layouts::{Data, DataMut, DataRef, ReaderFrom, ToOwnedDeep, VecZnx, VecZnxToMut, VecZnxToRef, WriterTo},
    source::Source,
};

use crate::layouts::{Infos, SetMetaData};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GLWECiphertext<D: Data> {
    pub data: VecZnx<D>,
    pub basek: usize,
    pub k: usize,
}

impl<D: DataRef> ToOwnedDeep for GLWECiphertext<D> {
    type Owned = GLWECiphertext<Vec<u8>>;
    fn to_owned_deep(&self) -> Self::Owned {
        GLWECiphertext {
            data: self.data.to_owned_deep(),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl<D: DataRef> fmt::Debug for GLWECiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataRef> fmt::Display for GLWECiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GLWECiphertext: basek={} k={}: {}",
            self.basek(),
            self.k(),
            self.data
        )
    }
}

impl<D: DataMut> Reset for GLWECiphertext<D>
where
    VecZnx<D>: Reset,
{
    fn reset(&mut self) {
        self.data.reset();
        self.basek = 0;
        self.k = 0;
    }
}

impl<D: DataMut> FillUniform for GLWECiphertext<D> {
    fn fill_uniform(&mut self, source: &mut Source) {
        self.data.fill_uniform(source);
    }
}

impl GLWECiphertext<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: VecZnx::alloc(n, rank + 1, k.div_ceil(basek)),
            basek,
            k,
        }
    }

    pub fn bytes_of(n: usize, basek: usize, k: usize, rank: usize) -> usize {
        VecZnx::alloc_bytes(n, rank + 1, k.div_ceil(basek))
    }
}

impl<D: Data> Infos for GLWECiphertext<D> {
    type Inner = VecZnx<D>;

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

impl<D: Data> GLWECiphertext<D> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<D: DataMut> SetMetaData for GLWECiphertext<D> {
    fn set_k(&mut self, k: usize) {
        self.k = k
    }

    fn set_basek(&mut self, basek: usize) {
        self.basek = basek
    }
}

pub trait GLWECiphertextToRef: Infos {
    fn to_ref(&self) -> GLWECiphertext<&[u8]>;
}

impl<D: DataRef> GLWECiphertextToRef for GLWECiphertext<D> {
    fn to_ref(&self) -> GLWECiphertext<&[u8]> {
        GLWECiphertext {
            data: self.data.to_ref(),
            basek: self.basek,
            k: self.k,
        }
    }
}

pub trait GLWECiphertextToMut: Infos {
    fn to_mut(&mut self) -> GLWECiphertext<&mut [u8]>;
}

impl<D: DataMut> GLWECiphertextToMut for GLWECiphertext<D> {
    fn to_mut(&mut self) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext {
            data: self.data.to_mut(),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl<D: DataMut> ReaderFrom for GLWECiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWECiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        self.data.write_to(writer)
    }
}
