use std::fmt;

use poulpy_hal::{
    api::ZnFillUniform,
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo, Zn, ZnxInfos, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::layouts::{Base2K, Degree, LWECiphertext, LWEInfos, TorusPrecision, compressed::Decompress};

#[derive(PartialEq, Eq, Clone)]
pub struct LWECiphertextCompressed<D: Data> {
    pub(crate) data: Zn<D>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) seed: [u8; 32],
}

impl<D: Data> LWEInfos for LWECiphertextCompressed<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: DataRef> fmt::Debug for LWECiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef> fmt::Display for LWECiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LWECiphertextCompressed: base2k={} k={} seed={:?}: {}",
            self.base2k(),
            self.k(),
            self.seed,
            self.data
        )
    }
}

impl<D: DataMut> FillUniform for LWECiphertextCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl LWECiphertextCompressed<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: LWEInfos,
    {
        Self::alloc_with(infos.base2k(), infos.k())
    }

    pub fn alloc_with(base2k: Base2K, k: TorusPrecision) -> Self {
        Self {
            data: Zn::alloc(1, 1, k.0.div_ceil(base2k.0) as usize),
            k,
            base2k,
            seed: [0u8; 32],
        }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: LWEInfos,
    {
        Self::alloc_bytes_with(infos.base2k(), infos.k())
    }

    pub fn alloc_bytes_with(base2k: Base2K, k: TorusPrecision) -> usize {
        Zn::alloc_bytes(1, 1, k.0.div_ceil(base2k.0) as usize)
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut> ReaderFrom for LWECiphertextCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        reader.read_exact(&mut self.seed)?;
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWECiphertextCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_all(&self.seed)?;
        self.data.write_to(writer)
    }
}

impl<D: DataMut, B: Backend, DR: DataRef> Decompress<B, LWECiphertextCompressed<DR>> for LWECiphertext<D>
where
    Module<B>: ZnFillUniform,
{
    fn decompress(&mut self, module: &Module<B>, other: &LWECiphertextCompressed<DR>) {
        debug_assert_eq!(self.size(), other.size());
        let mut source: Source = Source::new(other.seed);
        module.zn_fill_uniform(
            self.n().into(),
            other.base2k().into(),
            &mut self.data,
            0,
            &mut source,
        );
        (0..self.size()).for_each(|i| {
            self.data.at_mut(0, i)[0] = other.data.at(0, i)[0];
        });
    }
}
