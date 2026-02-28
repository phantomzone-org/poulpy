use std::collections::HashMap;

use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWELayout, GGLWEPrepared, GGLWEPreparedFactory, GGLWEPreparedToMut,
    GGLWEPreparedToRef, GGLWEToRef, GLWEAutomorphismKeyHelper, GLWEInfos, GetGaloisElement, LWEInfos, Rank, SetGaloisElement,
    TorusPrecision,
};

impl<K, BE: Backend> GLWEAutomorphismKeyHelper<K, BE> for HashMap<i64, K>
where
    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
{
    fn get_automorphism_key(&self, k: i64) -> Option<&K> {
        self.get(&k)
    }

    fn automorphism_key_infos(&self) -> GGLWELayout {
        self.get(self.keys().next().unwrap()).unwrap().gglwe_layout()
    }
}

#[derive(PartialEq, Eq)]
pub struct GLWEAutomorphismKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GGLWEPrepared<D, B>,
    pub(crate) p: i64,
}

impl<D: Data, B: Backend> LWEInfos for GLWEAutomorphismKeyPrepared<D, B> {
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

impl<D: Data, B: Backend> GetGaloisElement for GLWEAutomorphismKeyPrepared<D, B> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, B: Backend> SetGaloisElement for GLWEAutomorphismKeyPrepared<D, B> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}

impl<D: Data, B: Backend> GLWEInfos for GLWEAutomorphismKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GLWEAutomorphismKeyPrepared<D, B> {
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

pub trait GLWEAutomorphismKeyPreparedFactory<B: Backend>
where
    Self: GGLWEPreparedFactory<B>,
{
    fn alloc_glwe_automorphism_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GLWEAutomorphismKeyPrepared<Vec<u8>, B> {
        GLWEAutomorphismKeyPrepared::<Vec<u8>, B> {
            key: self.alloc_gglwe_prepared(base2k, k, rank, rank, dnum, dsize),
            p: 0,
        }
    }

    fn alloc_glwe_automorphism_key_prepared_from_infos<A>(&self, infos: &A) -> GLWEAutomorphismKeyPrepared<Vec<u8>, B>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for AutomorphismKeyPrepared"
        );
        self.alloc_glwe_automorphism_key_prepared(infos.base2k(), infos.k(), infos.rank(), infos.dnum(), infos.dsize())
    }

    fn bytes_of_glwe_automorphism_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        self.bytes_of_gglwe_prepared(base2k, k, rank, rank, dnum, dsize)
    }

    fn bytes_of_glwe_automorphism_key_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for AutomorphismKeyPrepared"
        );
        self.bytes_of_glwe_automorphism_key_prepared(infos.base2k(), infos.k(), infos.rank(), infos.dnum(), infos.dsize())
    }

    fn prepare_glwe_automorphism_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.prepare_gglwe_tmp_bytes(infos);
        lvl_0
    }

    fn prepare_glwe_automorphism_key<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGLWEPreparedToMut<B> + SetGaloisElement,
        O: GGLWEToRef + GetGaloisElement,
        Scratch<B>: ScratchAvailable,
    {
        let res_infos = res.to_mut();
        assert!(
            scratch.available() >= self.prepare_glwe_automorphism_key_tmp_bytes(&res_infos),
            "scratch.available(): {} < GLWEAutomorphismKeyPreparedFactory::prepare_glwe_automorphism_key_tmp_bytes: {}",
            scratch.available(),
            self.prepare_glwe_automorphism_key_tmp_bytes(&res_infos)
        );
        self.prepare_gglwe(res, other, scratch);
        res.set_p(other.p());
    }
}

impl<B: Backend> GLWEAutomorphismKeyPreparedFactory<B> for Module<B> where Module<B>: GGLWEPreparedFactory<B> {}

impl<B: Backend> GLWEAutomorphismKeyPrepared<Vec<u8>, B> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: GLWEAutomorphismKeyPreparedFactory<B>,
    {
        module.alloc_glwe_automorphism_key_prepared_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self
    where
        M: GLWEAutomorphismKeyPreparedFactory<B>,
    {
        module.alloc_glwe_automorphism_key_prepared(base2k, k, rank, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWEAutomorphismKeyPreparedFactory<B>,
    {
        module.bytes_of_glwe_automorphism_key_prepared_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize
    where
        M: GLWEAutomorphismKeyPreparedFactory<B>,
    {
        module.bytes_of_glwe_automorphism_key_prepared(base2k, k, rank, dnum, dsize)
    }
}

impl<B: Backend> GLWEAutomorphismKeyPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<M>(&self, module: &M) -> usize
    where
        M: GLWEAutomorphismKeyPreparedFactory<B>,
    {
        module.prepare_glwe_automorphism_key_tmp_bytes(self)
    }
}

impl<D: DataMut, B: Backend> GLWEAutomorphismKeyPrepared<D, B> {
    pub fn prepare<O, M>(&mut self, module: &M, other: &O, scratch: &mut Scratch<B>)
    where
        O: GGLWEToRef + GetGaloisElement,
        M: GLWEAutomorphismKeyPreparedFactory<B>,
        Scratch<B>: ScratchAvailable,
    {
        module.prepare_glwe_automorphism_key(self, other, scratch);
    }
}

impl<D: DataMut, B: Backend> GGLWEPreparedToMut<B> for GLWEAutomorphismKeyPrepared<D, B> {
    fn to_mut(&mut self) -> GGLWEPrepared<&mut [u8], B> {
        self.key.to_mut()
    }
}

impl<D: DataRef, BE: Backend> GGLWEPreparedToRef<BE> for GLWEAutomorphismKeyPrepared<D, BE> {
    fn to_ref(&self) -> GGLWEPrepared<&[u8], BE> {
        self.key.to_ref()
    }
}
