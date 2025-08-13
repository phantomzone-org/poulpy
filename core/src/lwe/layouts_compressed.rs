use std::fmt;

use backend::hal::{
    api::{FillUniform, Reset, VecZnxFillUniform, ZnxInfos, ZnxView, ZnxViewMut},
    layouts::{Backend, Data, DataMut, DataRef, MatZnx, Module, ReaderFrom, VecZnx, WriterTo},
};
use sampling::source::Source;

use crate::{
    Decompress, GGLWEEncryptSkFamily, GLWESwitchingKeyCompressed, GLWEToLWESwitchingKey, Infos, LWECiphertext, LWESwitchingKey,
    LWEToGLWESwitchingKey, SetMetaData,
};

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
            k: k,
            basek: basek,
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
        reader.read(&mut self.seed)?;
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

impl<D: DataMut, B: Backend, DR: DataRef> Decompress<B, LWECiphertextCompressed<DR>> for LWECiphertext<D> {
    fn decompress(&mut self, module: &Module<B>, other: &LWECiphertextCompressed<DR>)
    where
        Module<B>: VecZnxFillUniform,
    {
        let mut source = Source::new(other.seed);
        module.vec_znx_fill_uniform(other.basek(), &mut self.data, 0, other.k(), &mut source);
        (0..self.size()).for_each(|i| {
            self.data.at_mut(0, i)[0] = other.data.at(0, i)[0];
        });
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GLWEToLWESwitchingKeyCompressed<D: Data>(pub(crate) GLWESwitchingKeyCompressed<D>);

impl<D: DataRef> fmt::Debug for GLWEToLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for GLWEToLWESwitchingKeyCompressed<D> {
    fn fill_uniform(&mut self, source: &mut sampling::source::Source) {
        self.0.fill_uniform(source);
    }
}

impl<D: DataMut> Reset for GLWEToLWESwitchingKeyCompressed<D> {
    fn reset(&mut self) {
        self.0.reset();
    }
}

impl<D: DataRef> fmt::Display for GLWEToLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(GLWEToLWESwitchingKeyCompressed) {}", self.0)
    }
}

impl<D: Data> Infos for GLWEToLWESwitchingKeyCompressed<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.0.inner()
    }

    fn basek(&self) -> usize {
        self.0.basek()
    }

    fn k(&self) -> usize {
        self.0.k()
    }
}

impl<D: Data> GLWEToLWESwitchingKeyCompressed<D> {
    pub fn digits(&self) -> usize {
        self.0.digits()
    }

    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    pub fn rank_in(&self) -> usize {
        self.0.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.0.rank_out()
    }
}

impl<D: DataMut> ReaderFrom for GLWEToLWESwitchingKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWEToLWESwitchingKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl GLWEToLWESwitchingKeyCompressed<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize, rank_in: usize) -> Self {
        Self(GLWESwitchingKeyCompressed::alloc(
            n, basek, k, rows, 1, rank_in, 1,
        ))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize, rank_in: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        GLWEToLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k, rank_in)
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct LWEToGLWESwitchingKeyCompressed<D: Data>(pub(crate) GLWESwitchingKeyCompressed<D>);

impl<D: DataRef> fmt::Debug for LWEToGLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for LWEToGLWESwitchingKeyCompressed<D> {
    fn fill_uniform(&mut self, source: &mut sampling::source::Source) {
        self.0.fill_uniform(source);
    }
}

impl<D: DataMut> Reset for LWEToGLWESwitchingKeyCompressed<D> {
    fn reset(&mut self) {
        self.0.reset();
    }
}

impl<D: DataRef> fmt::Display for LWEToGLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(LWEToGLWESwitchingKeyCompressed) {}", self.0)
    }
}

impl<D: Data> Infos for LWEToGLWESwitchingKeyCompressed<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.0.inner()
    }

    fn basek(&self) -> usize {
        self.0.basek()
    }

    fn k(&self) -> usize {
        self.0.k()
    }
}

impl<D: Data> LWEToGLWESwitchingKeyCompressed<D> {
    pub fn digits(&self) -> usize {
        self.0.digits()
    }

    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    pub fn rank_in(&self) -> usize {
        self.0.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.0.rank_out()
    }
}

impl<D: DataMut> ReaderFrom for LWEToGLWESwitchingKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWEToGLWESwitchingKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl LWEToGLWESwitchingKeyCompressed<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize, rank_out: usize) -> Self {
        Self(GLWESwitchingKeyCompressed::alloc(
            n, basek, k, rows, 1, 1, rank_out,
        ))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize, rank_out: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        LWEToGLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k, rank_out)
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct LWESwitchingKeyCompressed<D: Data>(pub(crate) GLWESwitchingKeyCompressed<D>);

impl<D: DataRef> fmt::Debug for LWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for LWESwitchingKeyCompressed<D> {
    fn fill_uniform(&mut self, source: &mut sampling::source::Source) {
        self.0.fill_uniform(source);
    }
}

impl<D: DataMut> Reset for LWESwitchingKeyCompressed<D> {
    fn reset(&mut self) {
        self.0.reset();
    }
}

impl<D: DataRef> fmt::Display for LWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(LWESwitchingKeyCompressed) {}", self.0)
    }
}

impl<D: Data> Infos for LWESwitchingKeyCompressed<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.0.inner()
    }

    fn basek(&self) -> usize {
        self.0.basek()
    }

    fn k(&self) -> usize {
        self.0.k()
    }
}

impl<D: Data> LWESwitchingKeyCompressed<D> {
    pub fn digits(&self) -> usize {
        self.0.digits()
    }

    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    pub fn rank_in(&self) -> usize {
        self.0.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.0.rank_out()
    }
}

impl<D: DataMut> ReaderFrom for LWESwitchingKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWESwitchingKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl LWESwitchingKeyCompressed<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize) -> Self {
        Self(GLWESwitchingKeyCompressed::alloc(
            n, basek, k, rows, 1, 1, 1,
        ))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        LWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k)
    }
}
