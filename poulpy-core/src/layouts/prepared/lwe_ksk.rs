use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, LWESwitchingKeyToRef, Rank, TorusPrecision,
    prepared::{
        GLWESwitchingKeyPrepare, GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedAlloc, GLWESwitchingKeyPreparedToMut,
        GLWESwitchingKeyPreparedToRef,
    },
};

#[derive(PartialEq, Eq)]
pub struct LWESwitchingKeyPrepared<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for LWESwitchingKeyPrepared<D, B> {
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
impl<D: Data, B: Backend> GLWEInfos for LWESwitchingKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for LWESwitchingKeyPrepared<D, B> {
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

pub trait LWESwitchingKeyPreparedAlloc<B: Backend>
where
    Self: GLWESwitchingKeyPreparedAlloc<B>,
{
    fn alloc_lwe_switching_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        dnum: Dnum,
    ) -> LWESwitchingKeyPrepared<Vec<u8>, B> {
        LWESwitchingKeyPrepared(self.alloc_glwe_switching_key_prepared(base2k, k, Rank(1), Rank(1), dnum, Dsize(1)))
    }

    fn lwe_switching_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> LWESwitchingKeyPrepared<Vec<u8>, B>
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWESwitchingKey"
        );
        debug_assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWESwitchingKey"
        );
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for LWESwitchingKey"
        );
        self.alloc_lwe_switching_key_prepared(infos.base2k(), infos.k(), infos.dnum())
    }

    fn lwe_switching_key_prepared_bytes_of(&self, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> usize {
        self.bytes_of_glwe_switching_key_prepared(base2k, k, Rank(1), Rank(1), dnum, Dsize(1))
    }

    fn lwe_switching_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWESwitchingKey"
        );
        debug_assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWESwitchingKey"
        );
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for LWESwitchingKey"
        );
        self.lwe_switching_key_prepared_bytes_of(infos.base2k(), infos.k(), infos.dnum())
    }
}

impl<B: Backend> LWESwitchingKeyPreparedAlloc<B> for Module<B> where Self: GLWESwitchingKeyPreparedAlloc<B> {}

impl<B: Backend> LWESwitchingKeyPrepared<Vec<u8>, B>
where
    Module<B>: LWESwitchingKeyPreparedAlloc<B>,
{
    pub fn alloc_from_infos<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        module.lwe_switching_key_prepared_alloc_from_infos(infos)
    }

    pub fn alloc(module: &Module<B>, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> Self {
        module.alloc_lwe_switching_key_prepared(base2k, k, dnum)
    }

    pub fn bytes_of_from_infos<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.lwe_switching_key_prepared_bytes_of_from_infos(infos)
    }

    pub fn bytes_of(module: &Module<B>, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> usize {
        module.lwe_switching_key_prepared_bytes_of(base2k, k, dnum)
    }
}

pub trait LWESwitchingKeyPrepare<B: Backend>
where
    Self: GLWESwitchingKeyPrepare<B>,
{
    fn lwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A)
    where
        A: GGLWEInfos,
    {
        self.prepare_glwe_switching_key_tmp_bytes(infos);
    }
    fn lwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: LWESwitchingKeyPreparedToMut<B>,
        O: LWESwitchingKeyToRef,
    {
        self.prepare_glwe_switching(&mut res.to_mut().0, &other.to_ref().0, scratch);
    }
}

impl<B: Backend> LWESwitchingKeyPrepare<B> for Module<B> where Self: GLWESwitchingKeyPrepare<B> {}

impl<B: Backend> LWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<A>(&self, module: &Module<B>, infos: &A)
    where
        A: GGLWEInfos,
        Module<B>: LWESwitchingKeyPrepare<B>,
    {
        module.lwe_switching_key_prepare_tmp_bytes(infos);
    }
}

impl<D: DataMut, B: Backend> LWESwitchingKeyPrepared<D, B> {
    fn prepare<O>(&mut self, module: &Module<B>, other: &O, scratch: &mut Scratch<B>)
    where
        O: LWESwitchingKeyToRef,
        Module<B>: LWESwitchingKeyPrepare<B>,
    {
        module.lwe_switching_key_prepare(self, other, scratch);
    }
}

pub trait LWESwitchingKeyPreparedToRef<B: Backend> {
    fn to_ref(&self) -> LWESwitchingKeyPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> LWESwitchingKeyPreparedToRef<B> for LWESwitchingKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToRef<B>,
{
    fn to_ref(&self) -> LWESwitchingKeyPrepared<&[u8], B> {
        LWESwitchingKeyPrepared(self.0.to_ref())
    }
}

pub trait LWESwitchingKeyPreparedToMut<B: Backend> {
    fn to_mut(&mut self) -> LWESwitchingKeyPrepared<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> LWESwitchingKeyPreparedToMut<B> for LWESwitchingKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToMut<B>,
{
    fn to_mut(&mut self) -> LWESwitchingKeyPrepared<&mut [u8], B> {
        LWESwitchingKeyPrepared(self.0.to_mut())
    }
}
