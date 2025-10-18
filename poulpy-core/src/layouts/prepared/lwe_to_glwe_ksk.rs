use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, LWEToGLWESwitchingKeyToRef, Rank, TorusPrecision,
    prepared::{
        GLWESwitchingKeyPrepare, GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedAlloc, GLWESwitchingKeyPreparedToMut,
        GLWESwitchingKeyPreparedToRef,
    },
};

/// A special [GLWESwitchingKey] required to for the conversion from [LWE] to [GLWE].
#[derive(PartialEq, Eq)]
pub struct LWEToGLWESwitchingKeyPrepared<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for LWEToGLWESwitchingKeyPrepared<D, B> {
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

impl<D: Data, B: Backend> GLWEInfos for LWEToGLWESwitchingKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for LWEToGLWESwitchingKeyPrepared<D, B> {
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

pub trait LWEToGLWESwitchingKeyPreparedAlloc<B: Backend>
where
    Self: GLWESwitchingKeyPreparedAlloc<B>,
{
    fn alloc_lwe_to_glwe_switching_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_out: Rank,
        dnum: Dnum,
    ) -> LWEToGLWESwitchingKeyPrepared<Vec<u8>, B> {
        LWEToGLWESwitchingKeyPrepared(self.alloc_glwe_switching_key_prepared(base2k, k, Rank(1), rank_out, dnum, Dsize(1)))
    }
    fn alloc_lwe_to_glwe_switching_key_prepared_from_infos<A>(&self, infos: &A) -> LWEToGLWESwitchingKeyPrepared<Vec<u8>, B>
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWEToGLWESwitchingKey"
        );
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWEToGLWESwitchingKey"
        );
        self.alloc_lwe_to_glwe_switching_key_prepared(infos.base2k(), infos.k(), infos.rank_out(), infos.dnum())
    }

    fn bytes_of_lwe_to_glwe_switching_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_out: Rank,
        dnum: Dnum,
    ) -> usize {
        self.bytes_of_glwe_switching_key_prepared(base2k, k, Rank(1), rank_out, dnum, Dsize(1))
    }

    fn bytes_of_lwe_to_glwe_switching_key_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWEToGLWESwitchingKey"
        );
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWEToGLWESwitchingKey"
        );
        self.bytes_of_lwe_to_glwe_switching_key_prepared(infos.base2k(), infos.k(), infos.rank_out(), infos.dnum())
    }
}

impl<B: Backend> LWEToGLWESwitchingKeyPreparedAlloc<B> for Module<B> where Self: GLWESwitchingKeyPreparedAlloc<B> {}

impl<B: Backend> LWEToGLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: LWEToGLWESwitchingKeyPreparedAlloc<B>,
    {
        module.alloc_lwe_to_glwe_switching_key_prepared_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> Self
    where
        M: LWEToGLWESwitchingKeyPreparedAlloc<B>,
    {
        module.alloc_lwe_to_glwe_switching_key_prepared(base2k, k, rank_out, dnum)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: LWEToGLWESwitchingKeyPreparedAlloc<B>,
    {
        module.bytes_of_lwe_to_glwe_switching_key_prepared_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> usize
    where
        M: LWEToGLWESwitchingKeyPreparedAlloc<B>,
    {
        module.bytes_of_lwe_to_glwe_switching_key_prepared(base2k, k, rank_out, dnum)
    }
}

pub trait LWEToGLWESwitchingKeyPrepare<B: Backend>
where
    Self: GLWESwitchingKeyPrepare<B>,
{
    fn prepare_lwe_to_glwe_switching_key_tmp_bytes<A>(&self, infos: &A)
    where
        A: GGLWEInfos,
    {
        self.prepare_glwe_switching_key_tmp_bytes(infos);
    }

    fn prepare_lwe_to_glwe_switching_key<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: LWEToGLWESwitchingKeyPreparedToMut<B>,
        O: LWEToGLWESwitchingKeyToRef,
    {
        self.prepare_glwe_switching(&mut res.to_mut().0, &other.to_ref().0, scratch);
    }
}

impl<B: Backend> LWEToGLWESwitchingKeyPrepare<B> for Module<B> where Self: GLWESwitchingKeyPrepare<B> {}

impl<B: Backend> LWEToGLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<A, M>(&self, module: &M, infos: &A)
    where
        A: GGLWEInfos,
        M: LWEToGLWESwitchingKeyPrepare<B>,
    {
        module.prepare_lwe_to_glwe_switching_key_tmp_bytes(infos);
    }
}

impl<D: DataMut, B: Backend> LWEToGLWESwitchingKeyPrepared<D, B> {
    pub fn prepare<O, M>(&mut self, module: &M, other: &O, scratch: &mut Scratch<B>)
    where
        O: LWEToGLWESwitchingKeyToRef,
        M: LWEToGLWESwitchingKeyPrepare<B>,
    {
        module.prepare_lwe_to_glwe_switching_key(self, other, scratch);
    }
}

pub trait LWEToGLWESwitchingKeyPreparedToRef<B: Backend> {
    fn to_ref(&self) -> LWEToGLWESwitchingKeyPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> LWEToGLWESwitchingKeyPreparedToRef<B> for LWEToGLWESwitchingKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToRef<B>,
{
    fn to_ref(&self) -> LWEToGLWESwitchingKeyPrepared<&[u8], B> {
        LWEToGLWESwitchingKeyPrepared(self.0.to_ref())
    }
}

pub trait LWEToGLWESwitchingKeyPreparedToMut<B: Backend> {
    fn to_mut(&mut self) -> LWEToGLWESwitchingKeyPrepared<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> LWEToGLWESwitchingKeyPreparedToMut<B> for LWEToGLWESwitchingKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToMut<B>,
{
    fn to_mut(&mut self) -> LWEToGLWESwitchingKeyPrepared<&mut [u8], B> {
        LWEToGLWESwitchingKeyPrepared(self.0.to_mut())
    }
}
