use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWECompressed, GGLWECompressedToMut, GGLWECompressedToRef, GGLWEInfos, GGLWEToMut, GLWEInfos,
    GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, LWEInfos, LWEToGLWESwitchingKey, Rank, TorusPrecision,
    compressed::{GLWESwitchingKeyCompressed, GLWESwitchingKeyDecompress},
};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct LWEToGLWESwitchingKeyCompressed<D: Data>(pub(crate) GLWESwitchingKeyCompressed<D>);

impl<D: Data> LWEInfos for LWEToGLWESwitchingKeyCompressed<D> {
    fn n(&self) -> Degree {
        self.0.n()
    }

    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.0.k()
    }
    fn size(&self) -> usize {
        self.0.size()
    }
}
impl<D: Data> GLWEInfos for LWEToGLWESwitchingKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for LWEToGLWESwitchingKeyCompressed<D> {
    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

impl<D: DataRef> fmt::Debug for LWEToGLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for LWEToGLWESwitchingKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for LWEToGLWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(LWEToGLWESwitchingKeyCompressed) {}", self.0)
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
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWEToGLWESwitchingKeyCompressed"
        );
        assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWEToGLWESwitchingKeyCompressed"
        );
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank_out(),
            infos.dnum(),
        )
    }

    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> Self {
        LWEToGLWESwitchingKeyCompressed(GLWESwitchingKeyCompressed::alloc(
            n,
            base2k,
            k,
            Rank(1),
            rank_out,
            dnum,
            Dsize(1),
        ))
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWEToGLWESwitchingKeyCompressed"
        );
        assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWEToGLWESwitchingKeyCompressed"
        );
        GLWESwitchingKeyCompressed::bytes_of_from_infos(infos)
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> usize {
        GLWESwitchingKeyCompressed::bytes_of(n, base2k, k, Rank(1), dnum, Dsize(1))
    }
}

pub trait LWEToGLWESwitchingKeyDecompress
where
    Self: GLWESwitchingKeyDecompress,
{
    fn decompress_lwe_to_glwe_switching_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToRef + GLWESwitchingKeyDegrees,
    {
        self.decompress_glwe_switching_key(res, other);
    }
}

impl<B: Backend> LWEToGLWESwitchingKeyDecompress for Module<B> where Self: GLWESwitchingKeyDecompress {}

impl<D: DataMut> LWEToGLWESwitchingKey<D> {
    pub fn decompress<O, M>(&mut self, module: &M, other: &O)
    where
        O: GGLWECompressedToRef + GLWESwitchingKeyDegrees,
        M: LWEToGLWESwitchingKeyDecompress,
    {
        module.decompress_lwe_to_glwe_switching_key(self, other);
    }
}

impl<D: DataRef> GGLWECompressedToRef for LWEToGLWESwitchingKeyCompressed<D> {
    fn to_ref(&self) -> GGLWECompressed<&[u8]> {
        self.0.to_ref()
    }
}

impl<D: DataMut> GGLWECompressedToMut for LWEToGLWESwitchingKeyCompressed<D> {
    fn to_mut(&mut self) -> GGLWECompressed<&mut [u8]> {
        self.0.to_mut()
    }
}
