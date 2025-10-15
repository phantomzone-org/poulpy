use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, GLWESwitchingKey, GLWESwitchingKeyAlloc, GLWESwitchingKeyToMut,
    GLWESwitchingKeyToRef, LWEInfos, Rank, TorusPrecision,
};

use std::fmt;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWEToLWEKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank_in: Rank,
    pub dnum: Dnum,
}

impl LWEInfos for GLWEToLWEKeyLayout {
    fn n(&self) -> Degree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl GLWEInfos for GLWEToLWEKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for GLWEToLWEKeyLayout {
    fn rank_in(&self) -> Rank {
        self.rank_in
    }

    fn dsize(&self) -> Dsize {
        Dsize(1)
    }

    fn rank_out(&self) -> Rank {
        Rank(1)
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

/// A special [GLWESwitchingKey] required to for the conversion from [GLWECiphertext] to [LWECiphertext].
#[derive(PartialEq, Eq, Clone)]
pub struct GLWEToLWESwitchingKey<D: Data>(pub(crate) GLWESwitchingKey<D>);

impl<D: Data> LWEInfos for GLWEToLWESwitchingKey<D> {
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

impl<D: Data> GLWEInfos for GLWEToLWESwitchingKey<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}
impl<D: Data> GGLWEInfos for GLWEToLWESwitchingKey<D> {
    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

impl<D: DataRef> fmt::Debug for GLWEToLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GLWEToLWESwitchingKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for GLWEToLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(GLWEToLWESwitchingKey) {}", self.0)
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

pub trait GLWEToLWESwitchingKeyAlloc
where
    Self: GLWESwitchingKeyAlloc,
{
    fn alloc_glwe_to_lwe_switching_key(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        dnum: Dnum,
    ) -> GLWEToLWESwitchingKey<Vec<u8>> {
        GLWEToLWESwitchingKey(self.alloc_glwe_switching_key(base2k, k, rank_in, Rank(1), dnum, Dsize(1)))
    }

    fn alloc_glwe_to_lwe_switching_key_from_infos<A>(&self, infos: &A) -> GLWEToLWESwitchingKey<Vec<u8>>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for GLWEToLWESwitchingKey"
        );
        assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for GLWEToLWESwitchingKey"
        );
        self.alloc_glwe_to_lwe_switching_key(infos.base2k(), infos.k(), infos.rank_in(), infos.dnum())
    }

    fn bytes_of_glwe_to_lwe_switching_key(&self, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum) -> usize {
        self.bytes_of_glwe_switching_key(base2k, k, rank_in, Rank(1), dnum, Dsize(1))
    }

    fn bytes_of_glwe_to_lwe_switching_key_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for GLWEToLWESwitchingKey"
        );
        assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for GLWEToLWESwitchingKey"
        );
        self.bytes_of_glwe_to_lwe_switching_key(infos.base2k(), infos.k(), infos.rank_in(), infos.dnum())
    }
}

impl<B: Backend> GLWEToLWESwitchingKeyAlloc for Module<B> where Self: GLWESwitchingKeyAlloc {}

impl GLWEToLWESwitchingKey<Vec<u8>> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: GLWEToLWESwitchingKeyAlloc,
    {
        module.alloc_glwe_to_lwe_switching_key_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum) -> Self
    where
        M: GLWEToLWESwitchingKeyAlloc,
    {
        module.alloc_glwe_to_lwe_switching_key(base2k, k, rank_in, dnum)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWEToLWESwitchingKeyAlloc,
    {
        module.bytes_of_glwe_to_lwe_switching_key_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum) -> usize
    where
        M: GLWEToLWESwitchingKeyAlloc,
    {
        module.bytes_of_glwe_to_lwe_switching_key(base2k, k, rank_in, dnum)
    }
}

pub trait GLWEToLWESwitchingKeyToRef {
    fn to_ref(&self) -> GLWEToLWESwitchingKey<&[u8]>;
}

impl<D: DataRef> GLWEToLWESwitchingKeyToRef for GLWEToLWESwitchingKey<D>
where
    GLWESwitchingKey<D>: GLWESwitchingKeyToRef,
{
    fn to_ref(&self) -> GLWEToLWESwitchingKey<&[u8]> {
        GLWEToLWESwitchingKey(self.0.to_ref())
    }
}

pub trait GLWEToLWESwitchingKeyToMut {
    fn to_mut(&mut self) -> GLWEToLWESwitchingKey<&mut [u8]>;
}

impl<D: DataMut> GLWEToLWESwitchingKeyToMut for GLWEToLWESwitchingKey<D>
where
    GLWESwitchingKey<D>: GLWESwitchingKeyToMut,
{
    fn to_mut(&mut self) -> GLWEToLWESwitchingKey<&mut [u8]> {
        GLWEToLWESwitchingKey(self.0.to_mut())
    }
}
