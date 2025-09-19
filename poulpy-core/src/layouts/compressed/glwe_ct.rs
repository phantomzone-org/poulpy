use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, Reset, VecZnx, WriterTo},
    source::Source,
};

use crate::layouts::{GLWECiphertext, GLWEMetadata, Infos, compressed::Decompress};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GLWECiphertextCompressed<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) metadata: GLWEMetadata,
    pub(crate) seed: [u8; 32],
}

impl<D: Data> GLWECiphertextCompressed<D> {
    pub fn metadata(&self) -> GLWEMetadata {
        self.metadata
    }
}

impl<D: DataRef> fmt::Debug for GLWECiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef> fmt::Display for GLWECiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GLWECiphertextCompressed: basek={} k={} rank={} seed={:?}: {}",
            self.basek(),
            self.k(),
            self.rank(),
            self.seed,
            self.data
        )
    }
}

impl<D: DataMut> Reset for GLWECiphertextCompressed<D> {
    fn reset(&mut self) {
        self.data.reset();
        self.metadata.basek = 0;
        self.metadata.k = 0;
        self.metadata.rank = 0;
        self.seed = [0u8; 32];
    }
}

impl<D: DataMut> FillUniform for GLWECiphertextCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: Data> Infos for GLWECiphertextCompressed<D> {
    type Inner = VecZnx<D>;

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

impl<D: Data> GLWECiphertextCompressed<D> {
    pub fn rank(&self) -> usize {
        self.metadata.rank
    }
}

impl GLWECiphertextCompressed<Vec<u8>> {
    pub fn alloc(n: usize, metadata: GLWEMetadata) -> Self {
        Self {
            data: VecZnx::alloc(n, 1, metadata.k.div_ceil(metadata.basek)),
            metadata,
            seed: [0u8; 32],
        }
    }

    pub fn bytes_of(n: usize, basek: usize, k: usize) -> usize {
        GLWECiphertext::bytes_of(n, basek, k, 1)
    }
}

impl<D: DataMut> ReaderFrom for GLWECiphertextCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.metadata.k = reader.read_u64::<LittleEndian>()? as usize;
        self.metadata.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.metadata.rank = reader.read_u64::<LittleEndian>()? as usize;
        reader.read_exact(&mut self.seed)?;
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWECiphertextCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.metadata.k as u64)?;
        writer.write_u64::<LittleEndian>(self.metadata.basek as u64)?;
        writer.write_u64::<LittleEndian>(self.metadata.rank as u64)?;
        writer.write_all(&self.seed)?;
        self.data.write_to(writer)
    }
}

impl<D: DataMut, B: Backend, DR: DataRef> Decompress<B, GLWECiphertextCompressed<DR>> for GLWECiphertext<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &GLWECiphertextCompressed<DR>) {
        #[cfg(debug_assertions)]
        {
            use poulpy_hal::layouts::ZnxInfos;

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
        }

        let mut source: Source = Source::new(other.seed);
        self.decompress_internal(module, other, &mut source);
    }
}

impl<D: DataMut> GLWECiphertext<D> {
    pub(crate) fn decompress_internal<DataOther, B: Backend>(
        &mut self,
        module: &Module<B>,
        other: &GLWECiphertextCompressed<DataOther>,
        source: &mut Source,
    ) where
        DataOther: DataRef,
        Module<B>: VecZnxCopy + VecZnxFillUniform,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), other.rank());
            debug_assert_eq!(self.size(), other.size());
        }

        let k: usize = other.metadata.k;
        let basek: usize = other.metadata.basek;
        let cols: usize = other.rank() + 1;
        module.vec_znx_copy(&mut self.data, 0, &other.data, 0);
        (1..cols).for_each(|i| {
            module.vec_znx_fill_uniform(basek, &mut self.data, i, source);
        });

        self.metadata.basek = basek;
        self.metadata.k = k;
    }
}
