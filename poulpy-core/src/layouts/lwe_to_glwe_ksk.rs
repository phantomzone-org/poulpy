use std::fmt;

use poulpy_backend::hal::{
    api::{FillUniform, Reset},
    layouts::{Data, DataMut, DataRef, MatZnx, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{GGLWESwitchingKey, Infos};

#[derive(PartialEq, Eq, Clone)]
pub struct LWEToGLWESwitchingKey<D: Data>(pub(crate) GGLWESwitchingKey<D>);

impl<D: DataRef> fmt::Debug for LWEToGLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for LWEToGLWESwitchingKey<D> {
    fn fill_uniform(&mut self, source: &mut Source) {
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
        self.0.inner()
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
        Self(GGLWESwitchingKey::alloc(n, basek, k, rows, 1, 1, rank_out))
    }
}
