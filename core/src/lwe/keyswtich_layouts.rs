use backend::hal::{
    api::{FillUniform, Reset},
    layouts::{Backend, Data, DataMut, DataRef, MatZnx, Module, ReaderFrom, WriterTo},
};

use crate::{GGLWEEncryptSkFamily, GLWESecret, GLWESecretExec, GLWESwitchingKey, Infos};

use std::fmt;

/// A special [GLWESwitchingKey] required to for the conversion from [GLWECiphertext] to [LWECiphertext].
#[derive(PartialEq, Eq, Clone)]
pub struct GLWEToLWESwitchingKey<D: Data>(pub(crate) GLWESwitchingKey<D>);

impl<D: DataRef> fmt::Debug for GLWEToLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for GLWEToLWESwitchingKey<D> {
    fn fill_uniform(&mut self, source: &mut sampling::source::Source) {
        self.0.fill_uniform(source);
    }
}

impl<D: DataMut> Reset for GLWEToLWESwitchingKey<D> {
    fn reset(&mut self) {
        self.0.reset();
    }
}

impl<D: DataRef> fmt::Display for GLWEToLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(GLWEToLWESwitchingKey) {}", self.0)
    }
}

impl<D: Data> Infos for GLWEToLWESwitchingKey<D> {
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

impl<D: Data> GLWEToLWESwitchingKey<D> {
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

impl<D: DataMut> ReaderFrom for GLWEToLWESwitchingKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWEToLWESwitchingKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl GLWEToLWESwitchingKey<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize, rank_in: usize) -> Self {
        Self(GLWESwitchingKey::alloc(n, basek, k, rows, 1, rank_in, 1))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize, rank_in: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        GLWESecretExec::bytes_of(module, n, rank_in)
            + (GLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k, rank_in, 1) | GLWESecret::bytes_of(n, rank_in))
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct LWEToGLWESwitchingKey<D: Data>(pub(crate) GLWESwitchingKey<D>);

impl<D: DataRef> fmt::Debug for LWEToGLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for LWEToGLWESwitchingKey<D> {
    fn fill_uniform(&mut self, source: &mut sampling::source::Source) {
        self.0.fill_uniform(source);
    }
}

impl<D: DataMut> Reset for LWEToGLWESwitchingKey<D> {
    fn reset(&mut self) {
        self.0.reset();
    }
}

impl<D: DataRef> fmt::Display for LWEToGLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(LWEToGLWESwitchingKey) {}", self.0)
    }
}

impl<D: Data> Infos for LWEToGLWESwitchingKey<D> {
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

impl<D: Data> LWEToGLWESwitchingKey<D> {
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

impl<D: DataMut> ReaderFrom for LWEToGLWESwitchingKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWEToGLWESwitchingKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl LWEToGLWESwitchingKey<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize, rank_out: usize) -> Self {
        Self(GLWESwitchingKey::alloc(n, basek, k, rows, 1, 1, rank_out))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize, rank_out: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        GLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k, 1, rank_out) + GLWESecret::bytes_of(n, 1)
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct LWESwitchingKey<D: Data>(pub(crate) GLWESwitchingKey<D>);

impl<D: DataRef> fmt::Debug for LWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for LWESwitchingKey<D> {
    fn fill_uniform(&mut self, source: &mut sampling::source::Source) {
        self.0.fill_uniform(source);
    }
}

impl<D: DataMut> Reset for LWESwitchingKey<D> {
    fn reset(&mut self) {
        self.0.reset();
    }
}

impl<D: DataRef> fmt::Display for LWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(LWESwitchingKey) {}", self.0)
    }
}

impl<D: Data> Infos for LWESwitchingKey<D> {
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

impl<D: Data> LWESwitchingKey<D> {
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

impl<D: DataMut> ReaderFrom for LWESwitchingKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWESwitchingKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl LWESwitchingKey<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize) -> Self {
        Self(GLWESwitchingKey::alloc(n, basek, k, rows, 1, 1, 1))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        GLWESecret::bytes_of(n, 1)
            + GLWESecretExec::bytes_of(module, n, 1)
            + GLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k, 1, 1)
    }
}
