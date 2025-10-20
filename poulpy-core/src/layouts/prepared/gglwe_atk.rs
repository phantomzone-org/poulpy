use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEPrepare, GGLWEPrepared, GGLWEPreparedAlloc, GGLWEPreparedToMut,
    GGLWEPreparedToRef, GGLWEToRef, GLWEInfos, LWEInfos, Rank, TorusPrecision,
};

#[derive(PartialEq, Eq)]
pub struct AutomorphismKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GGLWEPrepared<D, B>,
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

pub trait AutomorphismKeyPreparedAlloc<B: Backend>
where
    Self: GGLWEPreparedAlloc<B>,
{
    fn alloc_automorphism_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> AutomorphismKeyPrepared<Vec<u8>, B> {
        AutomorphismKeyPrepared::<Vec<u8>, B> {
            key: self.alloc_gglwe_prepared(base2k, k, rank, rank, dnum, dsize),
            p: 0,
        }
    }

    fn alloc_automorphism_key_prepared_from_infos<A>(&self, infos: &A) -> AutomorphismKeyPrepared<Vec<u8>, B>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for AutomorphismKeyPrepared"
        );
        self.alloc_automorphism_key_prepared(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn bytes_of_automorphism_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        self.bytes_of_gglwe_prepared(base2k, k, rank, rank, dnum, dsize)
    }

    fn bytes_of_automorphism_key_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for AutomorphismKeyPrepared"
        );
        self.bytes_of_automorphism_key_prepared(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }
}

impl<B: Backend> AutomorphismKeyPreparedAlloc<B> for Module<B> where Module<B>: GGLWEPreparedAlloc<B> {}

impl<B: Backend> AutomorphismKeyPrepared<Vec<u8>, B> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: AutomorphismKeyPreparedAlloc<B>,
    {
        module.alloc_automorphism_key_prepared_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self
    where
        M: AutomorphismKeyPreparedAlloc<B>,
    {
        module.alloc_automorphism_key_prepared(base2k, k, rank, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: AutomorphismKeyPreparedAlloc<B>,
    {
        module.bytes_of_automorphism_key_prepared_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize
    where
        M: AutomorphismKeyPreparedAlloc<B>,
    {
        module.bytes_of_automorphism_key_prepared(base2k, k, rank, dnum, dsize)
    }
}

pub trait PrepareAutomorphismKey<B: Backend>
where
    Self: GGLWEPrepare<B>,
{
    fn prepare_automorphism_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.prepare_gglwe_tmp_bytes(infos)
    }

    fn prepare_automorphism_key<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGLWEPreparedToMut<B> + SetAutomorphismGaloisElement,
        O: GGLWEToRef + GetAutomorphismGaloisElement,
    {
        self.prepare_gglwe(res, other, scratch);
        res.set_p(other.p());
    }
}

impl<B: Backend> PrepareAutomorphismKey<B> for Module<B> where Module<B>: GGLWEPrepare<B> {}

impl<B: Backend> AutomorphismKeyPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<M>(&self, module: &M) -> usize
    where
        M: PrepareAutomorphismKey<B>,
    {
        module.prepare_automorphism_key_tmp_bytes(self)
    }
}

impl<D: DataMut, B: Backend> AutomorphismKeyPrepared<D, B> {
    pub fn prepare<O, M>(&mut self, module: &M, other: &O, scratch: &mut Scratch<B>)
    where
        O: GGLWEToRef + GetAutomorphismGaloisElement,
        M: PrepareAutomorphismKey<B>,
    {
        module.prepare_automorphism_key(self, other, scratch);
    }
}

impl<D: DataMut, B: Backend> GGLWEPreparedToMut<B> for AutomorphismKeyPrepared<D, B> {
    fn to_mut(&mut self) -> GGLWEPrepared<&mut [u8], B> {
        self.key.to_mut()
    }
}

impl<D: DataRef, BE: Backend> GGLWEPreparedToRef<BE> for AutomorphismKeyPrepared<D, BE> {
    fn to_ref(&self) -> GGLWEPrepared<&[u8], BE> {
        self.key.to_ref()
    }
}
