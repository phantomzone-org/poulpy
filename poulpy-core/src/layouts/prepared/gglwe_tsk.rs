use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, Rank, TensorKey, TensorKeyToRef, TorusPrecision,
    prepared::{
        GLWESwitchingKeyPrepare, GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedAlloc, GLWESwitchingKeyPreparedToMut,
        GLWESwitchingKeyPreparedToRef,
    },
};

#[derive(PartialEq, Eq)]
pub struct TensorKeyPrepared<D: Data, B: Backend> {
    pub(crate) keys: Vec<GLWESwitchingKeyPrepared<D, B>>,
}

impl<D: Data, B: Backend> LWEInfos for TensorKeyPrepared<D, B> {
    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.keys[0].k()
    }

    fn size(&self) -> usize {
        self.keys[0].size()
    }
}

impl<D: Data, B: Backend> GLWEInfos for TensorKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for TensorKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.rank_out()
    }

    fn rank_out(&self) -> Rank {
        self.keys[0].rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.keys[0].dsize()
    }

    fn dnum(&self) -> Dnum {
        self.keys[0].dnum()
    }
}

pub trait TensorKeyPreparedAlloc<B: Backend>
where
    Self: GLWESwitchingKeyPreparedAlloc<B>,
{
    fn tensor_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        dnum: Dnum,
        dsize: Dsize,
        rank: Rank,
    ) -> TensorKeyPrepared<Vec<u8>, B> {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        TensorKeyPrepared {
            keys: (0..pairs)
                .map(|_| self.glwe_switching_key_prepared_alloc(base2k, k, Rank(1), rank, dnum, dsize))
                .collect(),
        }
    }

    fn tensor_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> TensorKeyPrepared<Vec<u8>, B>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKeyPrepared"
        );
        self.tensor_key_prepared_alloc(
            infos.base2k(),
            infos.k(),
            infos.dnum(),
            infos.dsize(),
            infos.rank_out(),
        )
    }

    fn tensor_key_prepared_alloc_bytes(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let pairs: usize = (((rank.0 + 1) * rank.0) >> 1).max(1) as usize;
        pairs * self.glwe_switching_key_prepared_alloc_bytes(base2k, k, Rank(1), rank, dnum, dsize)
    }

    fn tensor_key_prepared_alloc_bytes_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.tensor_key_prepared_alloc_bytes(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }
}

impl<B: Backend> TensorKeyPreparedAlloc<B> for Module<B> where Module<B>: GLWESwitchingKeyPreparedAlloc<B> {}

impl<B: Backend> TensorKeyPrepared<Vec<u8>, B>
where
    Module<B>: TensorKeyPreparedAlloc<B>,
{
    pub fn alloc_from_infos<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        module.tensor_key_prepared_alloc_from_infos(infos)
    }

    pub fn alloc_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, dnum: Dnum, dsize: Dsize, rank: Rank) -> Self {
        module.tensor_key_prepared_alloc(base2k, k, dnum, dsize, rank)
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.tensor_key_prepared_alloc_bytes_from_infos(infos)
    }

    pub fn alloc_bytes_with(
        module: &Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        module.tensor_key_prepared_alloc_bytes(base2k, k, rank, dnum, dsize)
    }
}

impl<D: DataMut, B: Backend> TensorKeyPrepared<D, B> {
    // Returns a mutable reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GLWESwitchingKeyPrepared<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank_out().into();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataRef, B: Backend> TensorKeyPrepared<D, B> {
    // Returns a reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at(&self, mut i: usize, mut j: usize) -> &GLWESwitchingKeyPrepared<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank_out().into();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

pub trait TensorKeyPrepare<B: Backend>
where
    Self: GLWESwitchingKeyPrepare<B>,
{
    fn tensor_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.glwe_switching_key_prepare_tmp_bytes(infos)
    }

    fn tensor_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: TensorKeyPreparedToMut<B>,
        O: TensorKeyToRef,
    {
        let mut res: TensorKeyPrepared<&mut [u8], B> = res.to_mut();
        let other: TensorKey<&[u8]> = other.to_ref();

        assert_eq!(res.keys.len(), other.keys.len());

        for (a, b) in res.keys.iter_mut().zip(other.keys.iter()) {
            self.glwe_switching_prepare(a, b, scratch);
        }
    }
}

impl<B: Backend> TensorKeyPrepare<B> for Module<B> where Module<B>: GLWESwitchingKeyPrepare<B> {}

impl<B: Backend> TensorKeyPrepared<Vec<u8>, B>
where
    Module<B>: TensorKeyPrepare<B>,
{
    fn prepare_tmp_bytes<A>(&self, module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.tensor_key_prepare_tmp_bytes(infos)
    }
}

impl<D: DataMut, B: Backend> TensorKeyPrepared<D, B>
where
    Module<B>: TensorKeyPrepare<B>,
{
    fn prepare<O>(&mut self, module: &Module<B>, other: &O, scratch: &mut Scratch<B>)
    where
        O: TensorKeyToRef,
    {
        module.tensor_key_prepare(self, other, scratch);
    }
}

pub trait TensorKeyPreparedToMut<B: Backend> {
    fn to_mut(&mut self) -> TensorKeyPrepared<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> TensorKeyPreparedToMut<B> for TensorKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToMut<B>,
{
    fn to_mut(&mut self) -> TensorKeyPrepared<&mut [u8], B> {
        TensorKeyPrepared {
            keys: self.keys.iter_mut().map(|c| c.to_mut()).collect(),
        }
    }
}

pub trait TensorKeyPreparedToRef<B: Backend> {
    fn to_ref(&self) -> TensorKeyPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> TensorKeyPreparedToRef<B> for TensorKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToRef<B>,
{
    fn to_ref(&self) -> TensorKeyPrepared<&[u8], B> {
        TensorKeyPrepared {
            keys: self.keys.iter().map(|c| c.to_ref()).collect(),
        }
    }
}
