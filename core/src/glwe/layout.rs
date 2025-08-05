use backend::hal::{
    api::{VecZnxAlloc, VecZnxAllocBytes},
    layouts::{Backend, Module, ReaderFrom, VecZnx, VecZnxToMut, VecZnxToRef, WriterTo},
};

use crate::{GLWEOps, Infos, SetMetaData};

pub struct GLWECiphertext<D> {
    pub data: VecZnx<D>,
    pub basek: usize,
    pub k: usize,
}

impl GLWECiphertext<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> Self
    where
        Module<B>: VecZnxAlloc,
    {
        Self {
            data: module.vec_znx_alloc(rank + 1, k.div_ceil(basek)),
            basek,
            k,
        }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: VecZnxAllocBytes,
    {
        module.vec_znx_alloc_bytes(rank + 1, k.div_ceil(basek))
    }
}

impl<D> Infos for GLWECiphertext<D> {
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

impl<D> GLWECiphertext<D> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<DataSelf: AsRef<[u8]>> GLWECiphertext<DataSelf> {
    pub fn clone(&self) -> GLWECiphertext<Vec<u8>> {
        GLWECiphertext {
            data: self.data.clone(),
            basek: self.basek(),
            k: self.k(),
        }
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> SetMetaData for GLWECiphertext<DataSelf> {
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

impl<D: AsRef<[u8]>> GLWECiphertextToRef for GLWECiphertext<D> {
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

impl<D: AsMut<[u8]> + AsRef<[u8]>> GLWECiphertextToMut for GLWECiphertext<D> {
    fn to_mut(&mut self) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext {
            data: self.data.to_mut(),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl<D> GLWEOps for GLWECiphertext<D>
where
    D: AsRef<[u8]> + AsMut<[u8]>,
    GLWECiphertext<D>: GLWECiphertextToMut + Infos + SetMetaData,
{
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: AsRef<[u8]> + AsMut<[u8]>> ReaderFrom for GLWECiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.data.read_from(reader)
    }
}

impl<D: AsRef<[u8]>> WriterTo for GLWECiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        self.data.write_to(writer)
    }
}
