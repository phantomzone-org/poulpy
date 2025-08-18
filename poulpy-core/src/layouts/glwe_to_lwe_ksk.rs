use poulpy_hal::{
    api::{FillUniform, Reset},
    layouts::{Data, DataMut, DataRef, MatZnx, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{GGLWESwitchingKey, Infos};

use std::fmt;

/// A special [GLWESwitchingKey] required to for the conversion from [GLWECiphertext] to [LWECiphertext].
#[derive(PartialEq, Eq, Clone)]
pub struct GLWEToLWESwitchingKey<D: Data>(pub(crate) GGLWESwitchingKey<D>);

impl<D: DataRef> fmt::Debug for GLWEToLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for GLWEToLWESwitchingKey<D> {
    fn fill_uniform(&mut self, source: &mut Source) {
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
        self.0.inner()
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
        Self(GGLWESwitchingKey::alloc(n, basek, k, rows, 1, rank_in, 1))
    }
}
