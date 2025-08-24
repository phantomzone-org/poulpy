use poulpy_hal::{
    api::{
        SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal, VecZnxBigNormalize,
        VecZnxCopy, VecZnxDftAllocBytes, VecZnxDftFromVecZnx, VecZnxDftToVecZnxBigConsume, VecZnxFillUniform, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubABInplace,
    },
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, MatZnx, Module, ReaderFrom, Reset, WriterTo},
    source::Source,
};

use crate::layouts::{
    Infos, LWESwitchingKey,
    compressed::{Decompress, GGLWESwitchingKeyCompressed},
};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct LWESwitchingKeyCompressed<D: Data>(pub(crate) GGLWESwitchingKeyCompressed<D>);

impl<D: DataRef> fmt::Debug for LWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for LWESwitchingKeyCompressed<D> {
    fn fill_uniform(&mut self, source: &mut Source) {
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
        self.0.inner()
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
        Self(GGLWESwitchingKeyCompressed::alloc(
            n, basek, k, rows, 1, 1, 1,
        ))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> usize
    where
        Module<B>: VecZnxDftAllocBytes
            + VecZnxBigNormalize<B>
            + VecZnxDftFromVecZnx<B>
            + SvpApplyInplace<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxFillUniform
            + VecZnxSubABInplace
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxAddNormal
            + VecZnxNormalize<B>
            + VecZnxSub
            + SvpPrepare<B>
            + SvpPPolAllocBytes
            + SvpPPolAlloc<B>,
    {
        LWESwitchingKey::encrypt_sk_scratch_space(module, basek, k)
    }
}

impl<D: DataMut, DR: DataRef, B: Backend> Decompress<B, LWESwitchingKeyCompressed<DR>> for LWESwitchingKey<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &LWESwitchingKeyCompressed<DR>) {
        self.0.decompress(module, &other.0);
    }
}
