use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

use crate::layouts::{
    AutomorphismKeyToRef, Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    prepared::{
        GLWESwitchingKeyPrepare, GLWESwitchingKeyPrepareTmpBytes, GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedAlloc,
        GLWESwitchingKeyPreparedAllocBytes, GLWESwitchingKeyPreparedAllocBytesFromInfos, GLWESwitchingKeyPreparedAllocFromInfos,
        GLWESwitchingKeyPreparedToMut, GLWESwitchingKeyPreparedToRef,
    },
};

#[derive(PartialEq, Eq)]
pub struct AutomorphismKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GLWESwitchingKeyPrepared<D, B>,
    pub(crate) p: i64,
}

impl<D: Data, B: Backend> LWEInfos for AutomorphismKeyPrepared<D, B> {
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

pub trait GetAutomorphismGaloisElement {
    fn p(&self) -> i64;
}

impl<D: Data, B: Backend> GetAutomorphismGaloisElement for AutomorphismKeyPrepared<D, B> {
    fn p(&self) -> i64 {
        self.p
    }
}

pub trait SetAutomorphismGaloisElement {
    fn set_p(&mut self, p: i64);
}

impl<D: Data, B: Backend> SetAutomorphismGaloisElement for AutomorphismKeyPrepared<D, B> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}

impl<D: Data, B: Backend> GLWEInfos for AutomorphismKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for AutomorphismKeyPrepared<D, B> {
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

pub trait AutomorphismKeyPreparedAlloc<B: Backend> {
    fn automorphism_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> AutomorphismKeyPrepared<Vec<u8>, B>;
}

impl<B: Backend> AutomorphismKeyPreparedAlloc<B> for Module<B>
where
    Module<B>: GLWESwitchingKeyPreparedAlloc<B>,
{
    fn automorphism_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> AutomorphismKeyPrepared<Vec<u8>, B> {
        AutomorphismKeyPrepared::<Vec<u8>, B> {
            key: self.glwe_switching_key_prepared_alloc(base2k, k, rank, rank, dnum, dsize),
            p: 0,
        }
    }
}

pub trait AutomorphismKeyPreparedAllocFromInfos<B: Backend> {
    fn automorphism_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> AutomorphismKeyPrepared<Vec<u8>, B>
    where
        A: GGLWEInfos;
}

impl<B: Backend> AutomorphismKeyPreparedAllocFromInfos<B> for Module<B>
where
    Module<B>: GLWESwitchingKeyPreparedAllocFromInfos<B>,
{
    fn automorphism_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> AutomorphismKeyPrepared<Vec<u8>, B>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for AutomorphismKeyPrepared"
        );
        AutomorphismKeyPrepared {
            key: self.glwe_switching_key_prepared_alloc_from_infos(infos),
            p: 0,
        }
    }
}

pub trait AutomorphismKeyPreparedAllocBytes<B: Backend> {
    fn automorphism_key_prepared_alloc_bytes(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize;
}

impl<B: Backend> AutomorphismKeyPreparedAllocBytes<B> for Module<B>
where
    Module<B>: GLWESwitchingKeyPreparedAllocBytes<B>,
{
    fn automorphism_key_prepared_alloc_bytes(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        self.glwe_switching_key_prepared_alloc_bytes(base2k, k, rank, rank, dnum, dsize)
    }
}

pub trait AutomorphismKeyPreparedAllocBytesFromInfos<B: Backend> {
    fn automorphism_key_prepared_alloc_bytes_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;
}

impl<B: Backend> AutomorphismKeyPreparedAllocBytesFromInfos<B> for Module<B>
where
    Module<B>: GLWESwitchingKeyPreparedAllocBytesFromInfos<B>,
{
    fn automorphism_key_prepared_alloc_bytes_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKeyPrepared"
        );
        self.glwe_switching_key_prepared_alloc_bytes_from_infos(infos)
    }
}

impl<B: Backend> AutomorphismKeyPrepared<Vec<u8>, B> {
    pub fn alloc_from_infos<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: AutomorphismKeyPreparedAllocFromInfos<B>,
    {
        module.automorphism_key_prepared_alloc_from_infos(infos)
    }

    pub fn alloc_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self
    where
        Module<B>: AutomorphismKeyPreparedAlloc<B>,
    {
        module.automorphism_key_prepared_alloc(base2k, k, rank, dnum, dsize)
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: AutomorphismKeyPreparedAllocBytesFromInfos<B>,
    {
        module.automorphism_key_prepared_alloc_bytes_from_infos(infos)
    }

    pub fn alloc_bytes_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize
    where
        Module<B>: AutomorphismKeyPreparedAllocBytes<B>,
    {
        module.automorphism_key_prepared_alloc_bytes(base2k, k, rank, dnum, dsize)
    }
}

pub trait AutomorphismKeyPrepareTmpBytes {
    fn automorphism_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;
}

impl<B: Backend> AutomorphismKeyPrepareTmpBytes for Module<B>
where
    Module<B>: GLWESwitchingKeyPrepareTmpBytes,
{
    fn automorphism_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.glwe_switching_key_prepare_tmp_bytes(infos)
    }
}

impl<D: DataRef, B: Backend> AutomorphismKeyPrepared<D, B> {
    pub fn prepare_tmp_bytes(&self, module: &Module<B>) -> usize
    where
        Module<B>: AutomorphismKeyPrepareTmpBytes,
    {
        module.automorphism_key_prepare_tmp_bytes(self)
    }
}

pub trait AutomorphismKeyPrepare<B: Backend> {
    fn automorphism_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: AutomorphismKeyPreparedToMut<B> + SetAutomorphismGaloisElement,
        O: AutomorphismKeyToRef + GetAutomorphismGaloisElement;
}

impl<B: Backend> AutomorphismKeyPrepare<B> for Module<B>
where
    Module<B>: GLWESwitchingKeyPrepare<B>,
{
    fn automorphism_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: AutomorphismKeyPreparedToMut<B> + SetAutomorphismGaloisElement,
        O: AutomorphismKeyToRef + GetAutomorphismGaloisElement,
    {
        self.glwe_switching_prepare(&mut res.to_mut().key, &other.to_ref().key, scratch);
        res.set_p(other.p());
    }
}

pub trait AutomorphismKeyPreparedToMut<B: Backend> {
    fn to_mut(&mut self) -> AutomorphismKeyPrepared<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> AutomorphismKeyPreparedToMut<B> for AutomorphismKeyPrepared<D, B> {
    fn to_mut(&mut self) -> AutomorphismKeyPrepared<&mut [u8], B> {
        AutomorphismKeyPrepared {
            p: self.p,
            key: self.key.to_mut(),
        }
    }
}

pub trait AutomorphismKeyPreparedToRef<B: Backend> {
    fn to_ref(&self) -> AutomorphismKeyPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> AutomorphismKeyPreparedToRef<B> for AutomorphismKeyPrepared<D, B> {
    fn to_ref(&self) -> AutomorphismKeyPrepared<&[u8], B> {
        AutomorphismKeyPrepared {
            p: self.p,
            key: self.key.to_ref(),
        }
    }
}
