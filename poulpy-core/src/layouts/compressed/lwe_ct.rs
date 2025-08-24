use std::fmt;

use poulpy_hal::{
    api::ZnFillUniform,
    layouts::{
        Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, Reset, VecZnx, WriterTo, ZnxInfos, ZnxView, ZnxViewMut,
    },
    source::Source,
};

use crate::layouts::{Infos, LWECiphertext, SetMetaData, compressed::Decompress};

#[derive(PartialEq, Eq, Clone)]
pub struct LWECiphertextCompressed<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) k: usize,
    pub(crate) basek: usize,
    pub(crate) seed: [u8; 32],
}

impl<D: DataRef> fmt::Debug for LWECiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataRef> fmt::Display for LWECiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LWECiphertextCompressed: basek={} k={} seed={:?}: {}",
            self.basek(),
            self.k(),
            self.seed,
            self.data
        )
    }
}

impl<D: DataMut> Reset for LWECiphertextCompressed<D>
where
    VecZnx<D>: Reset,
{
    fn reset(&mut self) {
        self.data.reset();
        self.basek = 0;
        self.k = 0;
        self.seed = [0u8; 32];
    }
}

impl<D: DataMut> FillUniform for LWECiphertextCompressed<D> {
    fn fill_uniform(&mut self, source: &mut Source) {
        self.data.fill_uniform(source);
    }
}

impl LWECiphertextCompressed<Vec<u8>> {
    pub fn alloc(basek: usize, k: usize) -> Self {
        Self {
            data: VecZnx::alloc(1, 1, k.div_ceil(basek)),
            k,
            basek,
            seed: [0u8; 32],
        }
    }
}

impl<D: Data> Infos for LWECiphertextCompressed<D>
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

impl<DataSelf: DataMut> SetMetaData for LWECiphertextCompressed<DataSelf> {
    fn set_k(&mut self, k: usize) {
        self.k = k
    }

    fn set_basek(&mut self, basek: usize) {
        self.basek = basek
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut> ReaderFrom for LWECiphertextCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        reader.read_exact(&mut self.seed)?;
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWECiphertextCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        writer.write_all(&self.seed)?;
        self.data.write_to(writer)
    }
}

impl<D: DataMut, B: Backend, DR: DataRef> Decompress<B, LWECiphertextCompressed<DR>> for LWECiphertext<D>
where
    Module<B>: ZnFillUniform,
{
    fn decompress(&mut self, module: &Module<B>, other: &LWECiphertextCompressed<DR>) {
        let mut source: Source = Source::new(other.seed);
        module.zn_fill_uniform(
            self.n(),
            other.basek(),
            &mut self.data,
            0,
            other.k(),
            &mut source,
        );
        (0..self.size()).for_each(|i| {
            self.data.at_mut(0, i)[0] = other.data.at(0, i)[0];
        });
    }
}
