use backend::hal::{
    api::{VecZnxAlloc, VecZnxAllocBytes, ZnxViewMut, ZnxZero},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, VecZnx, VecZnxToMut, VecZnxToRef, WriterTo}, oep::VecZnxFillUniformImpl,
};
use bytemuck::cast_slice_mut;
use rand_core::RngCore;
use sampling::source::Source;
use std::fmt;
use std::io::Error;

use crate::{GLWEOps, Infos, SetMetaData};

#[derive(PartialEq, Eq, Clone)]
pub struct GLWECiphertext<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
}

impl<D: DataRef> fmt::Debug for GLWECiphertext<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<D: DataMut> ZnxZero for GLWECiphertext<D> {
    fn zero(&mut self) {
        self.data.zero();
    }

    fn zero_at(&mut self, i: usize, j: usize) {
        self.data.zero_at(i, j);
    }
}

impl<D: DataMut> FillUniform for GLWECiphertext<D> {
    fn fill_uniform(&mut self, source: &mut Source) {
        self.data.fill_uniform(source);
    }
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

impl<D: DataRef> GLWECiphertext<D> {
    pub fn clone(&self) -> GLWECiphertext<Vec<u8>> {
        GLWECiphertext {
            data: self.data.clone(),
            basek: self.basek(),
            k: self.k(),
        }
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

impl<D: DataMut> GLWEOps for GLWECiphertext<D> where GLWECiphertext<D>: GLWECiphertextToMut + Infos + SetMetaData {}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut, B: Backend> ReaderFrom<B> for GLWECiphertext<D> where B: VecZnxFillUniformImpl<B>{
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let tag: usize = reader.read_u8()? as usize;
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;

        match tag {
            0 => {
                self.data.read_from(reader) // regular deserialization
            }
            1 => {
                // Check that the receiver rank matches the serialized rank
                let rank: usize = reader.read_u64::<LittleEndian>()? as usize;
                if self.rank() != rank {
                    return Err(Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "Invalid ciphertext rank: self.rank() = {} but deserialized rank = {}",
                            self.rank(),
                            rank
                        ),
                    ));
                }

                // Read the seed
                let mut seed: [u8; 32] = [0u8; 32];
                reader.read_exact(&mut seed)?;

                // -----------------------------
                // Custom VecZnx Deserialization
                // -----------------------------

                // Check ring degree
                let n: usize = reader.read_u64::<LittleEndian>()? as usize;
                if self.n() != n {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("deserialized n={} != self.n()={}", n, self.n()),
                    ));
                }

                // Check columns (always 1 if compressed)
                let cols: usize = reader.read_u64::<LittleEndian>()? as usize;
                if cols != 1 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "deserialized cols={} != 1 (compressed ciphertext is expected to have 1 col)",
                            cols
                        ),
                    ));
                }

                // Check size
                let size: usize = reader.read_u64::<LittleEndian>()? as usize;
                if self.size() != size {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("deserialized size={} != self.size()={}", size, self.size()),
                    ));
                }

                // Check max size
                let max_size: usize = reader.read_u64::<LittleEndian>()? as usize;
                if self.data.max_size() != max_size {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "deserialized max_size={} != self.data.max_size()={}",
                            max_size,
                            self.data.max_size()
                        ),
                    ))?;
                }

                // Check length of bytes remaining
                let len: usize = reader.read_u64::<LittleEndian>()? as usize;
                let len_want: usize = VecZnx::<Vec<u8>>::alloc_bytes::<i64>(n, cols, size);
                if len != len_want {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "deserialized bytes len={} != VecZnx::<Vec<u8>>::alloc_bytes::<i64>(n={}, cols={}, size={})={}",
                            len, n, cols, size, len_want
                        ),
                    ));
                }

                // Deserialize the first column of the ciphertext
                for i in 0..size {
                    reader.read_i64_into::<LittleEndian>(self.data.at_mut(0, i))?;
                }

                // Populates the other columns of the ciphertext from seeded cprng
                let mut source: Source = Source::new(seed);
                (1..cols).for_each(|i|{
                    B::vec_znx_fill_uniform_impl(basek, &mut self.data, i, k, source);
                });

                Ok(())
            }
            _ => Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid ciphertext tag",
            )),
        }
    }
}

impl<D: DataRef> WriterTo for GLWECiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u8(0)?;
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        self.data.write_to(writer)
    }
}

pub struct GLWECiphertextSeeded<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) rank: usize,
    pub(crate) seed: [u8; 32],
}

impl GLWECiphertextSeeded<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> Self
    where
        Module<B>: VecZnxAlloc,
    {
        Self {
            data: module.vec_znx_alloc(1, k.div_ceil(basek)),
            basek,
            k,
            rank: rank,
            seed: [0u8; 32],
        }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> usize
    where
        Module<B>: VecZnxAllocBytes,
    {
        GLWECiphertext::bytes_of(module, basek, k, 1)
    }
}

impl<D: DataRef> WriterTo for GLWECiphertextSeeded<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u8(1)?;
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        writer.write_u64::<LittleEndian>(self.rank as u64)?;
        writer.write_all(&self.seed)?;
        self.data.write_to(writer)
    }
}
