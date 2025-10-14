use std::fmt;

use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, GLWESwitchingKey, GLWESwitchingKeyAlloc, GLWESwitchingKeyToMut,
    GLWESwitchingKeyToRef, LWEInfos, Rank, TorusPrecision,
};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct LWEToGLWESwitchingKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank_out: Rank,
    pub dnum: Dnum,
}

impl LWEInfos for LWEToGLWESwitchingKeyLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        self.n
    }
}

impl GLWEInfos for LWEToGLWESwitchingKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for LWEToGLWESwitchingKeyLayout {
    fn rank_in(&self) -> Rank {
        Rank(1)
    }

    fn dsize(&self) -> Dsize {
        Dsize(1)
    }

    fn rank_out(&self) -> Rank {
        self.rank_out
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct LWEToGLWESwitchingKey<D: Data>(pub(crate) GLWESwitchingKey<D>);

impl<D: Data> LWEInfos for LWEToGLWESwitchingKey<D> {
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

impl<D: Data> GLWEInfos for LWEToGLWESwitchingKey<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}
impl<D: Data> GGLWEInfos for LWEToGLWESwitchingKey<D> {
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

impl<D: DataRef> fmt::Debug for LWEToGLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for LWEToGLWESwitchingKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for LWEToGLWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(LWEToGLWESwitchingKey) {}", self.0)
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

pub trait LWEToGLWESwitchingKeyAlloc
where
    Self: GLWESwitchingKeyAlloc,
{
    fn alloc_lwe_to_glwe_switching_key(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_out: Rank,
        dnum: Dnum,
    ) -> LWEToGLWESwitchingKey<Vec<u8>> {
        LWEToGLWESwitchingKey(self.alloc_glwe_switching_key(base2k, k, Rank(1), rank_out, dnum, Dsize(1)))
    }

    fn alloc_lwe_to_glwe_switching_key_from_infos<A>(&self, infos: &A) -> LWEToGLWESwitchingKey<Vec<u8>>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWEToGLWESwitchingKey"
        );
        assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWEToGLWESwitchingKey"
        );

        self.alloc_lwe_to_glwe_switching_key(infos.base2k(), infos.k(), infos.rank_out(), infos.dnum())
    }

    fn bytes_of_lwe_to_glwe_switching_key(&self, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> usize {
        self.bytes_of_glwe_switching_key(base2k, k, Rank(1), rank_out, dnum, Dsize(1))
    }

    fn bytes_of_lwe_to_glwe_switching_key_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWEToGLWESwitchingKey"
        );
        assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWEToGLWESwitchingKey"
        );
        self.bytes_of_lwe_to_glwe_switching_key(infos.base2k(), infos.k(), infos.rank_out(), infos.dnum())
    }
}

impl<B: Backend> LWEToGLWESwitchingKeyAlloc for Module<B> where Self: GLWESwitchingKeyAlloc {}

impl LWEToGLWESwitchingKey<Vec<u8>> {
    pub fn alloc_from_infos<A, B: Backend>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: LWEToGLWESwitchingKeyAlloc,
    {
        module.alloc_lwe_to_glwe_switching_key_from_infos(infos)
    }

    pub fn alloc<B: Backend>(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> Self
    where
        Module<B>: LWEToGLWESwitchingKeyAlloc,
    {
        module.alloc_lwe_to_glwe_switching_key(base2k, k, rank_out, dnum)
    }

    pub fn bytes_of_from_infos<A, B: Backend>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: LWEToGLWESwitchingKeyAlloc,
    {
        module.bytes_of_lwe_to_glwe_switching_key_from_infos(infos)
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, base2k: Base2K, k: TorusPrecision, dnum: Dnum, rank_out: Rank) -> usize
    where
        Module<B>: LWEToGLWESwitchingKeyAlloc,
    {
        module.bytes_of_lwe_to_glwe_switching_key(base2k, k, rank_out, dnum)
    }
}

pub trait LWEToGLWESwitchingKeyToRef {
    fn to_ref(&self) -> LWEToGLWESwitchingKey<&[u8]>;
}

impl<D: DataRef> LWEToGLWESwitchingKeyToRef for LWEToGLWESwitchingKey<D>
where
    GLWESwitchingKey<D>: GLWESwitchingKeyToRef,
{
    fn to_ref(&self) -> LWEToGLWESwitchingKey<&[u8]> {
        LWEToGLWESwitchingKey(self.0.to_ref())
    }
}

pub trait LWEToGLWESwitchingKeyToMut {
    fn to_mut(&mut self) -> LWEToGLWESwitchingKey<&mut [u8]>;
}

impl<D: DataMut> LWEToGLWESwitchingKeyToMut for LWEToGLWESwitchingKey<D>
where
    GLWESwitchingKey<D>: GLWESwitchingKeyToMut,
{
    fn to_mut(&mut self) -> LWEToGLWESwitchingKey<&mut [u8]> {
        LWEToGLWESwitchingKey(self.0.to_mut())
    }
}
