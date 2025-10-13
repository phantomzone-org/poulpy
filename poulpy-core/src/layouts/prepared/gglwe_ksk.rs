use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, GLWESwitchingKeySetMetaData, GLWESwitchingKeyToRef,
    GLWESwtichingKeyGetMetaData, LWEInfos, Rank, TorusPrecision,
    prepared::{GGLWEPrepare, GGLWEPrepared, GGLWEPreparedAlloc, GGLWEPreparedToMut, GGLWEPreparedToRef},
};

#[derive(PartialEq, Eq)]
pub struct GLWESwitchingKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GGLWEPrepared<D, B>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

impl<D: DataMut, B: Backend> GLWESwitchingKeySetMetaData for GLWESwitchingKeyPrepared<D, B> {
    fn set_sk_in_n(&mut self, sk_in_n: usize) {
        self.sk_in_n = sk_in_n
    }

    fn set_sk_out_n(&mut self, sk_out_n: usize) {
        self.sk_out_n = sk_out_n
    }
}

impl<D: DataRef, B: Backend> GLWESwtichingKeyGetMetaData for GLWESwitchingKeyPrepared<D, B> {
    fn sk_in_n(&self) -> usize {
        self.sk_in_n
    }

    fn sk_out_n(&self) -> usize {
        self.sk_out_n
    }
}

impl<D: Data, B: Backend> LWEInfos for GLWESwitchingKeyPrepared<D, B> {
    fn n(&self) -> Degree {
        self.key.n()
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.key.k()
    }

    fn size(&self) -> usize {
        self.key.size()
    }
}

impl<D: Data, B: Backend> GLWEInfos for GLWESwitchingKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GLWESwitchingKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.key.dsize()
    }

    fn dnum(&self) -> Dnum {
        self.key.dnum()
    }
}

pub trait GLWESwitchingKeyPreparedAlloc<B: Backend>
where
    Self: GGLWEPreparedAlloc<B>,
{
    fn glwe_switching_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GLWESwitchingKeyPrepared<Vec<u8>, B> {
        GLWESwitchingKeyPrepared::<Vec<u8>, B> {
            key: self.gglwe_prepared_alloc(base2k, k, rank_in, rank_out, dnum, dsize),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    fn glwe_switching_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWESwitchingKeyPrepared<Vec<u8>, B>
    where
        A: GGLWEInfos,
    {
        self.glwe_switching_key_prepared_alloc(
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn glwe_switching_key_prepared_alloc_bytes(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        self.gglwe_prepared_alloc_bytes(base2k, k, rank_in, rank_out, dnum, dsize)
    }

    fn glwe_switching_key_prepared_alloc_bytes_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.glwe_switching_key_prepared_alloc_bytes(
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }
}

impl<B: Backend> GLWESwitchingKeyPreparedAlloc<B> for Module<B> where Module<B>: GGLWEPreparedAlloc<B> {}

impl<B: Backend> GLWESwitchingKeyPrepared<Vec<u8>, B>
where
    Module<B>: GLWESwitchingKeyPreparedAlloc<B>,
{
    pub fn alloc_from_infos<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        module.glwe_switching_key_prepared_alloc_from_infos(infos)
    }

    pub fn alloc(
        module: &Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> Self {
        module.glwe_switching_key_prepared_alloc(base2k, k, rank_in, rank_out, dnum, dsize)
    }

    pub fn alloc_bytes_from_infos<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.glwe_switching_key_prepared_alloc_bytes_from_infos(infos)
    }

    pub fn alloc_bytes(
        module: &Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        module.glwe_switching_key_prepared_alloc_bytes(base2k, k, rank_in, rank_out, dnum, dsize)
    }
}

pub trait GLWESwitchingKeyPrepare<B: Backend>
where
    Self: GGLWEPrepare<B>,
{
    fn glwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.gglwe_prepare_tmp_bytes(infos)
    }

    fn glwe_switching_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GLWESwitchingKeyPreparedToMut<B> + GLWESwitchingKeySetMetaData,
        O: GLWESwitchingKeyToRef + GLWESwtichingKeyGetMetaData,
    {
        self.gglwe_prepare(&mut res.to_mut().key, &other.to_ref().key, scratch);
        res.set_sk_in_n(other.sk_in_n());
        res.set_sk_out_n(other.sk_out_n());
    }
}

impl<B: Backend> GLWESwitchingKeyPrepare<B> for Module<B> where Self: GGLWEPrepare<B> {}

impl<D: DataMut, B: Backend> GLWESwitchingKeyPrepared<D, B> {
    pub fn prepare<O>(&mut self, module: &Module<B>, other: &O, scratch: &mut Scratch<B>)
    where
        O: GLWESwitchingKeyToRef + GLWESwtichingKeyGetMetaData,
        Module<B>: GLWESwitchingKeyPrepare<B>,
    {
        module.glwe_switching_prepare(self, other, scratch);
    }
}

impl<B: Backend> GLWESwitchingKeyPrepared<Vec<u8>, B>
where
    Module<B>: GLWESwitchingKeyPrepare<B>,
{
    pub fn prepare_tmp_bytes(&self, module: &Module<B>) -> usize {
        module.gglwe_prepare_tmp_bytes(self)
    }
}

pub trait GLWESwitchingKeyPreparedToMut<B: Backend> {
    fn to_mut(&mut self) -> GLWESwitchingKeyPrepared<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> GLWESwitchingKeyPreparedToMut<B> for GLWESwitchingKeyPrepared<D, B> {
    fn to_mut(&mut self) -> GLWESwitchingKeyPrepared<&mut [u8], B> {
        GLWESwitchingKeyPrepared {
            sk_in_n: self.sk_in_n,
            sk_out_n: self.sk_out_n,
            key: self.key.to_mut(),
        }
    }
}

pub trait GLWESwitchingKeyPreparedToRef<B: Backend> {
    fn to_ref(&self) -> GLWESwitchingKeyPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> GLWESwitchingKeyPreparedToRef<B> for GLWESwitchingKeyPrepared<D, B> {
    fn to_ref(&self) -> GLWESwitchingKeyPrepared<&[u8], B> {
        GLWESwitchingKeyPrepared {
            sk_in_n: self.sk_in_n,
            sk_out_n: self.sk_out_n,
            key: self.key.to_ref(),
        }
    }
}
