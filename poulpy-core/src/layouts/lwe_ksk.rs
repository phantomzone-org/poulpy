use std::fmt;

use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Dnum, Dsize, GGLWEInfos, GLWEInfos, GLWESwitchingKey, GLWESwitchingKeyAlloc, GLWESwitchingKeyToMut,
    GLWESwitchingKeyToRef, LWEInfos, Rank, RingDegree, TorusPrecision,
};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct LWESwitchingKeyLayout {
    pub n: RingDegree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub dnum: Dnum,
}

impl LWEInfos for LWESwitchingKeyLayout {
    fn n(&self) -> RingDegree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl GLWEInfos for LWESwitchingKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for LWESwitchingKeyLayout {
    fn rank_in(&self) -> Rank {
        Rank(1)
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

#[derive(PartialEq, Eq, Clone)]
pub struct LWESwitchingKey<D: Data>(pub(crate) GLWESwitchingKey<D>);

impl<D: Data> LWEInfos for LWESwitchingKey<D> {
    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.0.k()
    }

    fn n(&self) -> RingDegree {
        self.0.n()
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}

impl<D: Data> GLWEInfos for LWESwitchingKey<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for LWESwitchingKey<D> {
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

pub trait LWESwitchingKeyAlloc
where
    Self: GLWESwitchingKeyAlloc,
{
    fn alloc_lwe_switching_key(&self, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> LWESwitchingKey<Vec<u8>> {
        LWESwitchingKey(self.alloc_glwe_switching_key(base2k, k, Rank(1), Rank(1), dnum, Dsize(1)))
    }

    fn alloc_lwe_switching_key_from_infos<A>(&self, infos: &A) -> LWESwitchingKey<Vec<u8>>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWESwitchingKey"
        );
        assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWESwitchingKey"
        );
        assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for LWESwitchingKey"
        );
        self.alloc_lwe_switching_key(infos.base2k(), infos.k(), infos.dnum())
    }

    fn bytes_of_lwe_switching_key(&self, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> usize {
        self.bytes_of_glwe_switching_key(base2k, k, Rank(1), Rank(1), dnum, Dsize(1))
    }

    fn bytes_of_lwe_switching_key_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWESwitchingKey"
        );
        assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWESwitchingKey"
        );
        assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for LWESwitchingKey"
        );
        self.bytes_of_lwe_switching_key(infos.base2k(), infos.k(), infos.dnum())
    }
}

impl<B: Backend> LWESwitchingKeyAlloc for Module<B> where Self: GLWESwitchingKeyAlloc {}

impl LWESwitchingKey<Vec<u8>> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: LWESwitchingKeyAlloc,
    {
        module.alloc_lwe_switching_key_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> Self
    where
        M: LWESwitchingKeyAlloc,
    {
        module.alloc_lwe_switching_key(base2k, k, dnum)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: LWESwitchingKeyAlloc,
    {
        module.bytes_of_glwe_switching_key_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> usize
    where
        M: LWESwitchingKeyAlloc,
    {
        module.bytes_of_lwe_switching_key(base2k, k, dnum)
    }
}

impl<D: DataRef> fmt::Debug for LWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for LWESwitchingKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for LWESwitchingKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(LWESwitchingKey) {}", self.0)
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

pub trait LWESwitchingKeyToRef {
    fn to_ref(&self) -> LWESwitchingKey<&[u8]>;
}

impl<D: DataRef> LWESwitchingKeyToRef for LWESwitchingKey<D>
where
    GLWESwitchingKey<D>: GLWESwitchingKeyToRef,
{
    fn to_ref(&self) -> LWESwitchingKey<&[u8]> {
        LWESwitchingKey(self.0.to_ref())
    }
}

pub trait LWESwitchingKeyToMut {
    fn to_mut(&mut self) -> LWESwitchingKey<&mut [u8]>;
}

impl<D: DataMut> LWESwitchingKeyToMut for LWESwitchingKey<D>
where
    GLWESwitchingKey<D>: GLWESwitchingKeyToMut,
{
    fn to_mut(&mut self) -> LWESwitchingKey<&mut [u8]> {
        LWESwitchingKey(self.0.to_mut())
    }
}
