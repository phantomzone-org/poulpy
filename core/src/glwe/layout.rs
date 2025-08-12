use std::fmt::Debug;

use backend::hal::{
    api::{FillUniform, VecZnxAlloc, VecZnxAllocBytes, VecZnxCopy, VecZnxFillUniform, ZnxInfos, ZnxZero},
    layouts::{Backend, Data, DataMut, DataRef, Module, ReaderFrom, VecZnx, VecZnxToMut, VecZnxToRef, WriterTo},
};
use sampling::source::Source;

use crate::{GLWEOps, Infos, SetMetaData};

#[derive(PartialEq, Eq, Clone)]
pub struct GLWECiphertext<D: Data> {
    pub data: VecZnx<D>,
    pub basek: usize,
    pub k: usize,
}

impl<D: DataRef> Debug for GLWECiphertext<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GLWECiphertext: basek={} k={}: {}",
            self.basek(),
            self.k(),
            self.data
        )
    }
}

impl<D: DataMut> ZnxZero for GLWECiphertext<D>
where
    VecZnx<D>: ZnxZero,
{
    fn zero(&mut self) {
        self.data.zero()
    }

    fn zero_at(&mut self, i: usize, j: usize) {
        self.data.zero_at(i, j);
    }
}

impl<D: DataMut> FillUniform for GLWECiphertext<D>
where
    VecZnx<D>: FillUniform,
{
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

#[derive(PartialEq, Eq, Clone)]
pub struct GLWECiphertextCompressed<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) rank: usize,
    pub(crate) seed: [u8; 32],
}

impl<D: DataRef> Debug for GLWECiphertextCompressed<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GLWECiphertext: basek={} k={}: {}",
            self.basek(),
            self.k(),
            self.data
        )
    }
}

impl<D: DataMut> ZnxZero for GLWECiphertextCompressed<D>
where
    VecZnx<D>: ZnxZero,
{
    fn zero(&mut self) {
        self.data.zero()
    }

    fn zero_at(&mut self, i: usize, j: usize) {
        self.data.zero_at(i, j);
    }
}

impl<D: DataMut> FillUniform for GLWECiphertextCompressed<D>
where
    VecZnx<D>: FillUniform,
{
    fn fill_uniform(&mut self, source: &mut Source) {
        self.data.fill_uniform(source);
    }
}

impl<D: Data> Infos for GLWECiphertextCompressed<D> {
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

impl<D: Data> GLWECiphertextCompressed<D> {
    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl GLWECiphertextCompressed<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> Self
    where
        Module<B>: VecZnxAlloc,
    {
        Self {
            data: module.vec_znx_alloc(1, k.div_ceil(basek)),
            basek,
            k,
            rank,
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

impl<D: DataMut> ReaderFrom for GLWECiphertextCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.rank = reader.read_u64::<LittleEndian>()? as usize;
        reader.read(&mut self.seed)?;
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWECiphertextCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        writer.write_u64::<LittleEndian>(self.rank as u64)?;
        writer.write_all(&self.seed)?;
        self.data.write_to(writer)
    }
}

impl<D: DataMut> GLWECiphertext<D> {
    pub fn decompress<DataOther, B: Backend>(&mut self, module: &Module<B>, other: &GLWECiphertextCompressed<DataOther>)
    where
        DataOther: DataRef,
        Module<B>: VecZnxFillUniform + VecZnxCopy,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.n(),
                other.data.n(),
                "invalid receiver: self.n()={} != other.n()={}",
                self.n(),
                other.data.n()
            );
            assert_eq!(
                self.size(),
                other.size(),
                "invalid receiver: self.size()={} != other.size()={}",
                self.size(),
                other.size()
            );
            assert_eq!(
                self.rank(),
                other.rank(),
                "invalid receiver: self.rank()={} != other.rank()={}",
                self.rank(),
                other.rank()
            );
            let mut source: Source = Source::new(other.seed);
            self.decompress_internal(module, other, &mut source);
        }
    }

    pub(crate) fn decompress_internal<DataOther, B: Backend>(
        &mut self,
        module: &Module<B>,
        other: &GLWECiphertextCompressed<DataOther>,
        source: &mut Source,
    ) where
        DataOther: DataRef,
        Module<B>: VecZnxFillUniform + VecZnxCopy,
    {
        let k: usize = other.k;
        let basek: usize = other.basek;
        let cols: usize = other.cols();
        module.vec_znx_copy(&mut self.data, 0, &other.data, 0);
        (1..cols).for_each(|i| {
            module.vec_znx_fill_uniform(basek, &mut self.data, i, k, source);
        });
        self.basek = basek;
        self.k = k;
    }
}
