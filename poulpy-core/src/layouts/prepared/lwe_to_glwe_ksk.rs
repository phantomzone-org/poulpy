use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, LWEToGLWESwitchingKeyToRef, Rank, TorusPrecision,
    prepared::{
        GLWESwitchingKeyPrepare, GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedAlloc, GLWESwitchingKeyPreparedToMut,
        GLWESwitchingKeyPreparedToRef,
    },
};

/// A special [GLWESwitchingKey] required to for the conversion from [LWECiphertext] to [GLWECiphertext].
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
    fn lwe_to_glwe_switching_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_out: Rank,
        dnum: Dnum,
    ) -> LWEToGLWESwitchingKeyPrepared<Vec<u8>, B> {
        LWEToGLWESwitchingKeyPrepared(self.alloc_glwe_switching_key_prepared(base2k, k, Rank(1), rank_out, dnum, Dsize(1)))
    }
    fn lwe_to_glwe_switching_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> LWEToGLWESwitchingKeyPrepared<Vec<u8>, B>
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
        self.lwe_to_glwe_switching_key_prepared_alloc(infos.base2k(), infos.k(), infos.rank_out(), infos.dnum())
    }

    fn lwe_to_glwe_switching_key_prepared_bytes_of(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_out: Rank,
        dnum: Dnum,
    ) -> usize {
        self.bytes_of_glwe_switching_key_prepared(base2k, k, Rank(1), rank_out, dnum, Dsize(1))
    }

    fn lwe_to_glwe_switching_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
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
        self.lwe_to_glwe_switching_key_prepared_bytes_of(infos.base2k(), infos.k(), infos.rank_out(), infos.dnum())
    }
}

impl<B: Backend> LWEToGLWESwitchingKeyPreparedAlloc<B> for Module<B> where Self: GLWESwitchingKeyPreparedAlloc<B> {}

impl<B: Backend> LWEToGLWESwitchingKeyPrepared<Vec<u8>, B>
where
    Module<B>: LWEToGLWESwitchingKeyPreparedAlloc<B>,
{
    pub fn alloc_from_infos<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        module.lwe_to_glwe_switching_key_prepared_alloc_from_infos(infos)
    }

    pub fn alloc(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> Self {
        module.lwe_to_glwe_switching_key_prepared_alloc(base2k, k, rank_out, dnum)
    }

    pub fn bytes_of_from_infos<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.lwe_to_glwe_switching_key_prepared_bytes_of_from_infos(infos)
    }

    pub fn bytes_of(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> usize {
        module.lwe_to_glwe_switching_key_prepared_bytes_of(base2k, k, rank_out, dnum)
    }
}

pub trait LWEToGLWESwitchingKeyPrepare<B: Backend>
where
    Self: GLWESwitchingKeyPrepare<B>,
{
    fn lwe_to_glwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A)
    where
        A: GGLWEInfos,
    {
        self.prepare_glwe_switching_key_tmp_bytes(infos);
    }

    fn lwe_to_glwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: LWEToGLWESwitchingKeyPreparedToMut<B>,
        O: LWEToGLWESwitchingKeyToRef,
    {
        self.prepare_glwe_switching(&mut res.to_mut().0, &other.to_ref().0, scratch);
    }
}

impl<B: Backend> LWEToGLWESwitchingKeyPrepare<B> for Module<B> where Self: GLWESwitchingKeyPrepare<B> {}

impl<B: Backend> LWEToGLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<A>(&self, module: &Module<B>, infos: &A)
    where
        A: GGLWEInfos,
        Module<B>: LWEToGLWESwitchingKeyPrepare<B>,
    {
        module.lwe_to_glwe_switching_key_prepare_tmp_bytes(infos);
    }
}

impl<D: DataMut, B: Backend> LWEToGLWESwitchingKeyPrepared<D, B> {
    fn prepare<O>(&mut self, module: &Module<B>, other: &O, scratch: &mut Scratch<B>)
    where
        O: LWEToGLWESwitchingKeyToRef,
        Module<B>: LWEToGLWESwitchingKeyPrepare<B>,
    {
        module.lwe_to_glwe_switching_key_prepare(self, other, scratch);
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
