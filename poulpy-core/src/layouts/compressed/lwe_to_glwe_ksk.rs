use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, LWEToGLWESwitchingKey, LWEToGLWESwitchingKeyToMut, Rank, RingDegree,
    TorusPrecision,
    compressed::{
        GLWESwitchingKeyCompressed, GLWESwitchingKeyCompressedAlloc, GLWESwitchingKeyCompressedToMut,
        GLWESwitchingKeyCompressedToRef, GLWESwitchingKeyDecompress,
    },
};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct LWEToGLWESwitchingKeyCompressed<D: Data>(pub(crate) GLWESwitchingKeyCompressed<D>);

impl<D: Data> LWEInfos for LWEToGLWESwitchingKeyCompressed<D> {
    fn n(&self) -> RingDegree {
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

pub trait LWEToGLWESwitchingKeyCompressedAlloc
where
    Self: GLWESwitchingKeyCompressedAlloc,
{
    fn alloc_lwe_to_glwe_switching_key_compressed(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_out: Rank,
        dnum: Dnum,
    ) -> LWEToGLWESwitchingKeyCompressed<Vec<u8>> {
        LWEToGLWESwitchingKeyCompressed(self.alloc_glwe_switching_key_compressed(base2k, k, Rank(1), rank_out, dnum, Dsize(1)))
    }

    fn alloc_lwe_to_glwe_switching_key_compressed_from_infos<A>(&self, infos: &A) -> LWEToGLWESwitchingKeyCompressed<Vec<u8>>
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
        self.alloc_lwe_to_glwe_switching_key_compressed(infos.base2k(), infos.k(), infos.rank_out(), infos.dnum())
    }

    fn bytes_of_lwe_to_glwe_switching_key_compressed(&self, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> usize {
        self.bytes_of_glwe_switching_key_compressed(base2k, k, Rank(1), dnum, Dsize(1))
    }

    fn bytes_of_lwe_to_glwe_switching_key_compressed_from_infos<A>(&self, infos: &A) -> usize
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
        self.bytes_of_lwe_to_glwe_switching_key_compressed(infos.base2k(), infos.k(), infos.dnum())
    }
}

impl LWEToGLWESwitchingKeyCompressed<Vec<u8>> {
    pub fn alloc<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: LWEToGLWESwitchingKeyCompressedAlloc,
    {
        module.alloc_lwe_to_glwe_switching_key_compressed_from_infos(infos)
    }

    pub fn alloc_with<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> Self
    where
        M: LWEToGLWESwitchingKeyCompressedAlloc,
    {
        module.alloc_lwe_to_glwe_switching_key_compressed(base2k, k, rank_out, dnum)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: LWEToGLWESwitchingKeyCompressedAlloc,
    {
        module.bytes_of_lwe_to_glwe_switching_key_compressed_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> usize
    where
        M: LWEToGLWESwitchingKeyCompressedAlloc,
    {
        module.bytes_of_lwe_to_glwe_switching_key_compressed(base2k, k, dnum)
    }
}

pub trait LWEToGLWESwitchingKeyDecompress
where
    Self: GLWESwitchingKeyDecompress,
{
    fn decompress_lwe_to_glwe_switching_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: LWEToGLWESwitchingKeyToMut,
        O: LWEToGLWESwitchingKeyCompressedToRef,
    {
        self.decompress_glwe_switching_key(&mut res.to_mut().0, &other.to_ref().0);
    }
}

impl<B: Backend> LWEToGLWESwitchingKeyDecompress for Module<B> where Self: GLWESwitchingKeyDecompress {}

impl<D: DataMut> LWEToGLWESwitchingKey<D> {
    pub fn decompress<O, M>(&mut self, module: &M, other: &O)
    where
        O: LWEToGLWESwitchingKeyCompressedToRef,
        M: LWEToGLWESwitchingKeyDecompress,
    {
        module.decompress_lwe_to_glwe_switching_key(self, other);
    }
}

pub trait LWEToGLWESwitchingKeyCompressedToRef {
    fn to_ref(&self) -> LWEToGLWESwitchingKeyCompressed<&[u8]>;
}

impl<D: DataRef> LWEToGLWESwitchingKeyCompressedToRef for LWEToGLWESwitchingKeyCompressed<D>
where
    GLWESwitchingKeyCompressed<D>: GLWESwitchingKeyCompressedToRef,
{
    fn to_ref(&self) -> LWEToGLWESwitchingKeyCompressed<&[u8]> {
        LWEToGLWESwitchingKeyCompressed(self.0.to_ref())
    }
}

pub trait LWEToGLWESwitchingKeyCompressedToMut {
    fn to_mut(&mut self) -> LWEToGLWESwitchingKeyCompressed<&mut [u8]>;
}

impl<D: DataMut> LWEToGLWESwitchingKeyCompressedToMut for LWEToGLWESwitchingKeyCompressed<D>
where
    GLWESwitchingKeyCompressed<D>: GLWESwitchingKeyCompressedToMut,
{
    fn to_mut(&mut self) -> LWEToGLWESwitchingKeyCompressed<&mut [u8]> {
        LWEToGLWESwitchingKeyCompressed(self.0.to_mut())
    }
}
