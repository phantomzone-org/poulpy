use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, LWESwitchingKey, LWESwitchingKeyToMut, Rank, TorusPrecision,
    compressed::{
        GLWESwitchingKeyCompressed, GLWESwitchingKeyCompressedAlloc, GLWESwitchingKeyCompressedToMut,
        GLWESwitchingKeyCompressedToRef, GLWESwitchingKeyDecompress,
    },
};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct LWESwitchingKeyCompressed<D: Data>(pub(crate) GLWESwitchingKeyCompressed<D>);

impl<D: Data> LWEInfos for LWESwitchingKeyCompressed<D> {
    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.0.k()
    }

    fn n(&self) -> Degree {
        self.0.n()
    }
    fn size(&self) -> usize {
        self.0.size()
    }
}
impl<D: Data> GLWEInfos for LWESwitchingKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for LWESwitchingKeyCompressed<D> {
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

impl<D: DataRef> fmt::Debug for LWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for LWESwitchingKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for LWESwitchingKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(LWESwitchingKeyCompressed) {}", self.0)
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

pub trait LWESwitchingKeyCompressedAlloc
where
    Self: GLWESwitchingKeyCompressedAlloc,
{
    fn alloc_lwe_switching_key_compressed(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        dnum: Dnum,
    ) -> LWESwitchingKeyCompressed<Vec<u8>> {
        LWESwitchingKeyCompressed(self.alloc_glwe_switching_key_compressed(base2k, k, Rank(1), Rank(1), dnum, Dsize(1)))
    }

    fn alloc_lwe_switching_key_compressed_from_infos<A>(&self, infos: &A) -> LWESwitchingKeyCompressed<Vec<u8>>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWESwitchingKeyCompressed"
        );
        assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWESwitchingKeyCompressed"
        );
        assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for LWESwitchingKeyCompressed"
        );
        self.alloc_lwe_switching_key_compressed(infos.base2k(), infos.k(), infos.dnum())
    }

    fn bytes_of_lwe_switching_key_compressed(&self, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> usize {
        self.bytes_of_glwe_switching_key_compressed(base2k, k, Rank(1), dnum, Dsize(1))
    }

    fn bytes_of_lwe_switching_key_compressed_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWESwitchingKeyCompressed"
        );
        assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWESwitchingKeyCompressed"
        );
        assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for LWESwitchingKeyCompressed"
        );
        self.bytes_of_glwe_switching_key_compressed_from_infos(infos)
    }
}

impl<B: Backend> LWESwitchingKeyCompressedAlloc for Module<B> where Self: GLWESwitchingKeyCompressedAlloc {}

impl LWESwitchingKeyCompressed<Vec<u8>> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: LWESwitchingKeyCompressedAlloc,
    {
        module.alloc_lwe_switching_key_compressed_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> Self
    where
        M: LWESwitchingKeyCompressedAlloc,
    {
        module.alloc_lwe_switching_key_compressed(base2k, k, dnum)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: LWESwitchingKeyCompressedAlloc,
    {
        module.bytes_of_lwe_switching_key_compressed_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> usize
    where
        M: LWESwitchingKeyCompressedAlloc,
    {
        module.bytes_of_lwe_switching_key_compressed(base2k, k, dnum)
    }
}

pub trait LWESwitchingKeyDecompress
where
    Self: GLWESwitchingKeyDecompress,
{
    fn decompress_lwe_switching_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: LWESwitchingKeyToMut,
        O: LWESwitchingKeyCompressedToRef,
    {
        self.decompress_glwe_switching_key(&mut res.to_mut().0, &other.to_ref().0);
    }
}

impl<B: Backend> LWESwitchingKeyDecompress for Module<B> where Self: GLWESwitchingKeyDecompress {}

impl<D: DataMut> LWESwitchingKey<D> {
    pub fn decompress<O, M>(&mut self, module: &M, other: &O)
    where
        O: LWESwitchingKeyCompressedToRef,
        M: LWESwitchingKeyDecompress,
    {
        module.decompress_lwe_switching_key(self, other);
    }
}

pub trait LWESwitchingKeyCompressedToRef {
    fn to_ref(&self) -> LWESwitchingKeyCompressed<&[u8]>;
}

impl<D: DataRef> LWESwitchingKeyCompressedToRef for LWESwitchingKeyCompressed<D>
where
    GLWESwitchingKeyCompressed<D>: GLWESwitchingKeyCompressedToRef,
{
    fn to_ref(&self) -> LWESwitchingKeyCompressed<&[u8]> {
        LWESwitchingKeyCompressed(self.0.to_ref())
    }
}

pub trait LWESwitchingKeyCompressedToMut {
    fn to_mut(&mut self) -> LWESwitchingKeyCompressed<&mut [u8]>;
}

impl<D: DataMut> LWESwitchingKeyCompressedToMut for LWESwitchingKeyCompressed<D>
where
    GLWESwitchingKeyCompressed<D>: GLWESwitchingKeyCompressedToMut,
{
    fn to_mut(&mut self) -> LWESwitchingKeyCompressed<&mut [u8]> {
        LWESwitchingKeyCompressed(self.0.to_mut())
    }
}
