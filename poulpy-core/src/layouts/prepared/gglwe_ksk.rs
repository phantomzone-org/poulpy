use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEToRef, GLWEInfos, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut,
    LWEInfos, Rank, TorusPrecision,
    prepared::{GGLWEPrepare, GGLWEPrepared, GGLWEPreparedAlloc, GGLWEPreparedToMut, GGLWEPreparedToRef},
};

#[derive(PartialEq, Eq)]
pub struct GLWESwitchingKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GGLWEPrepared<D, B>,
    pub(crate) input_degree: Degree,  // Degree of sk_in
    pub(crate) output_degree: Degree, // Degree of sk_out
}

impl<D: DataRef, BE: Backend> GLWESwitchingKeyDegrees for GLWESwitchingKeyPrepared<D, BE> {
    fn output_degree(&self) -> &Degree {
        &self.output_degree
    }

    fn input_degree(&self) -> &Degree {
        &self.input_degree
    }
}

impl<D: DataMut, BE: Backend> GLWESwitchingKeyDegreesMut for GLWESwitchingKeyPrepared<D, BE> {
    fn output_degree(&mut self) -> &mut Degree {
        &mut self.output_degree
    }

    fn input_degree(&mut self) -> &mut Degree {
        &mut self.input_degree
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
    fn alloc_glwe_switching_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GLWESwitchingKeyPrepared<Vec<u8>, B> {
        GLWESwitchingKeyPrepared::<Vec<u8>, B> {
            key: self.alloc_gglwe_prepared(base2k, k, rank_in, rank_out, dnum, dsize),
            input_degree: Degree(0),
            output_degree: Degree(0),
        }
    }

    fn alloc_glwe_switching_key_prepared_from_infos<A>(&self, infos: &A) -> GLWESwitchingKeyPrepared<Vec<u8>, B>
    where
        A: GGLWEInfos,
    {
        self.alloc_glwe_switching_key_prepared(
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn bytes_of_glwe_switching_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        self.bytes_of_gglwe_prepared(base2k, k, rank_in, rank_out, dnum, dsize)
    }

    fn bytes_of_glwe_switching_key_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.bytes_of_glwe_switching_key_prepared(
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }
}

impl<B: Backend> GLWESwitchingKeyPreparedAlloc<B> for Module<B> where Self: GGLWEPreparedAlloc<B> {}

impl<B: Backend> GLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: GLWESwitchingKeyPreparedAlloc<B>,
    {
        module.alloc_glwe_switching_key_prepared_from_infos(infos)
    }

    pub fn alloc<M>(
        module: &M,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> Self
    where
        M: GLWESwitchingKeyPreparedAlloc<B>,
    {
        module.alloc_glwe_switching_key_prepared(base2k, k, rank_in, rank_out, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWESwitchingKeyPreparedAlloc<B>,
    {
        module.bytes_of_glwe_switching_key_prepared_from_infos(infos)
    }

    pub fn bytes_of<M>(
        module: &M,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize
    where
        M: GLWESwitchingKeyPreparedAlloc<B>,
    {
        module.bytes_of_glwe_switching_key_prepared(base2k, k, rank_in, rank_out, dnum, dsize)
    }
}

pub trait GLWESwitchingKeyPrepare<B: Backend>
where
    Self: GGLWEPrepare<B>,
{
    fn prepare_glwe_switching_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.prepare_gglwe_tmp_bytes(infos)
    }

    fn prepare_glwe_switching<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGLWEPreparedToMut<B> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToRef + GLWESwitchingKeyDegrees,
    {
        self.prepare_gglwe(res, other, scratch);
        *res.input_degree() = *other.input_degree();
        *res.output_degree() = *other.output_degree();
    }
}

impl<B: Backend> GLWESwitchingKeyPrepare<B> for Module<B> where Self: GGLWEPrepare<B> {}

impl<D: DataMut, B: Backend> GLWESwitchingKeyPrepared<D, B> {
    pub fn prepare<O, M>(&mut self, module: &M, other: &O, scratch: &mut Scratch<B>)
    where
        O: GGLWEToRef + GLWESwitchingKeyDegrees,
        M: GLWESwitchingKeyPrepare<B>,
    {
        module.prepare_glwe_switching(self, other, scratch);
    }
}

impl<B: Backend> GLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<M>(&self, module: &M) -> usize
    where
        M: GLWESwitchingKeyPrepare<B>,
    {
        module.prepare_glwe_switching_key_tmp_bytes(self)
    }
}

impl<D: DataRef, BE: Backend> GGLWEPreparedToRef<BE> for GLWESwitchingKeyPrepared<D, BE>
where
    GGLWEPrepared<D, BE>: GGLWEPreparedToRef<BE>,
{
    fn to_ref(&self) -> GGLWEPrepared<&[u8], BE> {
        self.key.to_ref()
    }
}

impl<D: DataRef, BE: Backend> GGLWEPreparedToMut<BE> for GLWESwitchingKeyPrepared<D, BE>
where
    GGLWEPrepared<D, BE>: GGLWEPreparedToMut<BE>,
{
    fn to_mut(&mut self) -> GGLWEPrepared<&mut [u8], BE> {
        self.key.to_mut()
    }
}
