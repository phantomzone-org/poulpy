use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedFactory, GGLWEPreparedToMut, GGLWEPreparedToRef,
    GGLWEToRef, GLWEInfos, LWEInfos, Rank, TorusPrecision,
};

#[derive(PartialEq, Eq)]
pub struct GLWETensorKeyPrepared<D: Data, B: Backend>(pub(crate) GGLWEPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for GLWETensorKeyPrepared<D, B> {
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

impl<D: Data, B: Backend> GLWEInfos for GLWETensorKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GLWETensorKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.rank_out()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

pub trait GLWETensorKeyPreparedFactory<B: Backend>
where
    Self: GGLWEPreparedFactory<B>,
{
    fn alloc_tensor_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        dnum: Dnum,
        dsize: Dsize,
        rank: Rank,
    ) -> GLWETensorKeyPrepared<Vec<u8>, B> {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        GLWETensorKeyPrepared(self.alloc_gglwe_prepared(base2k, k, Rank(pairs), rank, dnum, dsize))
    }

    fn alloc_tensor_key_prepared_from_infos<A>(&self, infos: &A) -> GLWETensorKeyPrepared<Vec<u8>, B>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for TensorKeyPrepared"
        );
        self.alloc_tensor_key_prepared(infos.base2k(), infos.k(), infos.dnum(), infos.dsize(), infos.rank_out())
    }

    fn bytes_of_tensor_key_prepared(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        self.bytes_of_gglwe_prepared(base2k, k, Rank(pairs), rank, dnum, dsize)
    }

    fn bytes_of_tensor_key_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.bytes_of_tensor_key_prepared(infos.base2k(), infos.k(), infos.rank(), infos.dnum(), infos.dsize())
    }

    fn prepare_tensor_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.prepare_gglwe_tmp_bytes(infos)
    }

    fn prepare_tensor_key<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGLWEPreparedToMut<B>,
        O: GGLWEToRef,
    {
        self.prepare_gglwe(res, other, scratch);
    }
}

impl<B: Backend> GLWETensorKeyPreparedFactory<B> for Module<B> where Module<B>: GGLWEPreparedFactory<B> {}

impl<B: Backend> GLWETensorKeyPrepared<Vec<u8>, B> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: GLWETensorKeyPreparedFactory<B>,
    {
        module.alloc_tensor_key_prepared_from_infos(infos)
    }

    pub fn alloc_with<M>(module: &M, base2k: Base2K, k: TorusPrecision, dnum: Dnum, dsize: Dsize, rank: Rank) -> Self
    where
        M: GLWETensorKeyPreparedFactory<B>,
    {
        module.alloc_tensor_key_prepared(base2k, k, dnum, dsize, rank)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWETensorKeyPreparedFactory<B>,
    {
        module.bytes_of_tensor_key_prepared_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize
    where
        M: GLWETensorKeyPreparedFactory<B>,
    {
        module.bytes_of_tensor_key_prepared(base2k, k, rank, dnum, dsize)
    }
}

impl<B: Backend> GLWETensorKeyPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<A, M>(&self, module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWETensorKeyPreparedFactory<B>,
    {
        module.prepare_tensor_key_tmp_bytes(infos)
    }
}

impl<D: DataMut, B: Backend> GLWETensorKeyPrepared<D, B> {
    pub fn prepare<O, M>(&mut self, module: &M, other: &O, scratch: &mut Scratch<B>)
    where
        O: GGLWEToRef,
        M: GLWETensorKeyPreparedFactory<B>,
    {
        module.prepare_tensor_key(self, other, scratch);
    }
}

impl<D: DataMut, B: Backend> GGLWEPreparedToMut<B> for GLWETensorKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToMut<B>,
{
    fn to_mut(&mut self) -> GGLWEPrepared<&mut [u8], B> {
        self.0.to_mut()
    }
}

impl<D: DataRef, B: Backend> GGLWEPreparedToRef<B> for GLWETensorKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToRef<B>,
{
    fn to_ref(&self) -> GGLWEPrepared<&[u8], B> {
        self.0.to_ref()
    }
}
