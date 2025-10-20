use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedToMut, GGLWEPreparedToRef, GGLWEToRef, GLWEInfos,
    GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, LWEInfos, Rank, TorusPrecision,
    prepared::{GLWESwitchingKeyPrepare, GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedAlloc},
};

#[derive(PartialEq, Eq)]
pub struct GLWEToLWESwitchingKeyPrepared<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for GLWEToLWESwitchingKeyPrepared<D, B> {
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

impl<D: Data, B: Backend> GLWEInfos for GLWEToLWESwitchingKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GLWEToLWESwitchingKeyPrepared<D, B> {
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

pub trait GLWEToLWESwitchingKeyPreparedAlloc<B: Backend>
where
    Self: GLWESwitchingKeyPreparedAlloc<B>,
{
    fn alloc_glwe_to_lwe_switching_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        dnum: Dnum,
    ) -> GLWEToLWESwitchingKeyPrepared<Vec<u8>, B> {
        GLWEToLWESwitchingKeyPrepared(self.alloc_glwe_switching_key_prepared(base2k, k, rank_in, Rank(1), dnum, Dsize(1)))
    }
    fn alloc_glwe_to_lwe_switching_key_prepared_from_infos<A>(&self, infos: &A) -> GLWEToLWESwitchingKeyPrepared<Vec<u8>, B>
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for GLWEToLWESwitchingKeyPrepared"
        );
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for GLWEToLWESwitchingKeyPrepared"
        );
        self.alloc_glwe_to_lwe_switching_key_prepared(infos.base2k(), infos.k(), infos.rank_in(), infos.dnum())
    }

    fn bytes_of_glwe_to_lwe_switching_key_prepared(&self, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum) -> usize {
        self.bytes_of_glwe_switching_key_prepared(base2k, k, rank_in, Rank(1), dnum, Dsize(1))
    }

    fn bytes_of_glwe_to_lwe_switching_key_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for GLWEToLWESwitchingKeyPrepared"
        );
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for GLWEToLWESwitchingKeyPrepared"
        );
        self.bytes_of_glwe_to_lwe_switching_key_prepared(infos.base2k(), infos.k(), infos.rank_in(), infos.dnum())
    }
}

impl<B: Backend> GLWEToLWESwitchingKeyPreparedAlloc<B> for Module<B> where Self: GLWESwitchingKeyPreparedAlloc<B> {}

impl<B: Backend> GLWEToLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: GLWEToLWESwitchingKeyPreparedAlloc<B>,
    {
        module.alloc_glwe_to_lwe_switching_key_prepared_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum) -> Self
    where
        M: GLWEToLWESwitchingKeyPreparedAlloc<B>,
    {
        module.alloc_glwe_to_lwe_switching_key_prepared(base2k, k, rank_in, dnum)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWEToLWESwitchingKeyPreparedAlloc<B>,
    {
        module.bytes_of_glwe_to_lwe_switching_key_prepared_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum) -> usize
    where
        M: GLWEToLWESwitchingKeyPreparedAlloc<B>,
    {
        module.bytes_of_glwe_to_lwe_switching_key_prepared(base2k, k, rank_in, dnum)
    }
}

pub trait GLWEToLWESwitchingKeyPrepare<B: Backend>
where
    Self: GLWESwitchingKeyPrepare<B>,
{
    fn prepare_glwe_to_lwe_switching_key_tmp_bytes<A>(&self, infos: &A)
    where
        A: GGLWEInfos,
    {
        self.prepare_glwe_switching_key_tmp_bytes(infos);
    }

    fn prepare_glwe_to_lwe_switching_key<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGLWEPreparedToMut<B> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToRef + GLWESwitchingKeyDegrees,
    {
        self.prepare_glwe_switching(res, other, scratch);
    }
}

impl<B: Backend> GLWEToLWESwitchingKeyPrepare<B> for Module<B> where Self: GLWESwitchingKeyPrepare<B> {}

impl<B: Backend> GLWEToLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<A, M>(&self, module: &M, infos: &A)
    where
        A: GGLWEInfos,
        M: GLWEToLWESwitchingKeyPrepare<B>,
    {
        module.prepare_glwe_to_lwe_switching_key_tmp_bytes(infos);
    }
}

impl<D: DataMut, B: Backend> GLWEToLWESwitchingKeyPrepared<D, B> {
    pub fn prepare<O, M>(&mut self, module: &M, other: &O, scratch: &mut Scratch<B>)
    where
        O: GGLWEToRef + GLWESwitchingKeyDegrees,
        M: GLWEToLWESwitchingKeyPrepare<B>,
    {
        module.prepare_glwe_to_lwe_switching_key(self, other, scratch);
    }
}

impl<D: DataRef, B: Backend> GGLWEPreparedToRef<B> for GLWEToLWESwitchingKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GGLWEPreparedToRef<B>,
{
    fn to_ref(&self) -> GGLWEPrepared<&[u8], B> {
        self.0.to_ref()
    }
}

impl<D: DataMut, B: Backend> GGLWEPreparedToMut<B> for GLWEToLWESwitchingKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GGLWEPreparedToRef<B>,
{
    fn to_mut(&mut self) -> GGLWEPrepared<&mut [u8], B> {
        self.0.to_mut()
    }
}

impl<D: DataMut, B: Backend> GLWESwitchingKeyDegreesMut for GLWEToLWESwitchingKeyPrepared<D, B> {
    fn input_degree(&mut self) -> &mut Degree {
        &mut self.0.input_degree
    }

    fn output_degree(&mut self) -> &mut Degree {
        &mut self.0.output_degree
    }
}
