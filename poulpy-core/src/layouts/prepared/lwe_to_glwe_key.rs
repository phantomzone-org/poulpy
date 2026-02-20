use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedToMut, GGLWEPreparedToRef, GGLWEToRef, GLWEInfos,
    GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, LWEInfos, Rank, TorusPrecision,
    prepared::{GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedFactory},
};

/// A special `GLWESwitchingKey` required for the conversion from `LWE` to `GLWE`.
#[derive(PartialEq, Eq)]
pub struct LWEToGLWEKeyPrepared<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for LWEToGLWEKeyPrepared<D, B> {
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

impl<D: Data, B: Backend> GLWEInfos for LWEToGLWEKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for LWEToGLWEKeyPrepared<D, B> {
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

pub trait LWEToGLWEKeyPreparedFactory<B: Backend>
where
    Self: GLWESwitchingKeyPreparedFactory<B>,
{
    fn alloc_lwe_to_glwe_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_out: Rank,
        dnum: Dnum,
    ) -> LWEToGLWEKeyPrepared<Vec<u8>, B> {
        LWEToGLWEKeyPrepared(self.alloc_glwe_switching_key_prepared(base2k, k, Rank(1), rank_out, dnum, Dsize(1)))
    }
    fn alloc_lwe_to_glwe_key_prepared_from_infos<A>(&self, infos: &A) -> LWEToGLWEKeyPrepared<Vec<u8>, B>
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWEToGLWEKey");
        debug_assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWEToGLWEKey");
        self.alloc_lwe_to_glwe_key_prepared(infos.base2k(), infos.k(), infos.rank_out(), infos.dnum())
    }

    fn bytes_of_lwe_to_glwe_key_prepared(&self, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> usize {
        self.bytes_of_glwe_key_prepared(base2k, k, Rank(1), rank_out, dnum, Dsize(1))
    }

    fn bytes_of_lwe_to_glwe_key_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWEToGLWEKey");
        debug_assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWEToGLWEKey");
        self.bytes_of_lwe_to_glwe_key_prepared(infos.base2k(), infos.k(), infos.rank_out(), infos.dnum())
    }

    fn prepare_lwe_to_glwe_key_tmp_bytes<A>(&self, infos: &A)
    where
        A: GGLWEInfos,
    {
        self.prepare_glwe_switching_key_tmp_bytes(infos);
    }

    fn prepare_lwe_to_glwe_key<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGLWEPreparedToMut<B> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToRef + GLWESwitchingKeyDegrees,
    {
        self.prepare_glwe_switching(res, other, scratch);
    }
}

impl<B: Backend> LWEToGLWEKeyPreparedFactory<B> for Module<B> where Self: GLWESwitchingKeyPreparedFactory<B> {}

impl<B: Backend> LWEToGLWEKeyPrepared<Vec<u8>, B> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: LWEToGLWEKeyPreparedFactory<B>,
    {
        module.alloc_lwe_to_glwe_key_prepared_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> Self
    where
        M: LWEToGLWEKeyPreparedFactory<B>,
    {
        module.alloc_lwe_to_glwe_key_prepared(base2k, k, rank_out, dnum)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: LWEToGLWEKeyPreparedFactory<B>,
    {
        module.bytes_of_lwe_to_glwe_key_prepared_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> usize
    where
        M: LWEToGLWEKeyPreparedFactory<B>,
    {
        module.bytes_of_lwe_to_glwe_key_prepared(base2k, k, rank_out, dnum)
    }
}

impl<B: Backend> LWEToGLWEKeyPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<A, M>(&self, module: &M, infos: &A)
    where
        A: GGLWEInfos,
        M: LWEToGLWEKeyPreparedFactory<B>,
    {
        module.prepare_lwe_to_glwe_key_tmp_bytes(infos);
    }
}

impl<D: DataMut, B: Backend> LWEToGLWEKeyPrepared<D, B> {
    pub fn prepare<O, M>(&mut self, module: &M, other: &O, scratch: &mut Scratch<B>)
    where
        O: GGLWEToRef + GLWESwitchingKeyDegrees,
        M: LWEToGLWEKeyPreparedFactory<B>,
    {
        module.prepare_lwe_to_glwe_key(self, other, scratch);
    }
}

impl<D: DataRef, B: Backend> GGLWEPreparedToRef<B> for LWEToGLWEKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GGLWEPreparedToRef<B>,
{
    fn to_ref(&self) -> GGLWEPrepared<&[u8], B> {
        self.0.to_ref()
    }
}

impl<D: DataMut, B: Backend> GGLWEPreparedToMut<B> for LWEToGLWEKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GGLWEPreparedToMut<B>,
{
    fn to_mut(&mut self) -> GGLWEPrepared<&mut [u8], B> {
        self.0.to_mut()
    }
}

impl<D: DataMut, B: Backend> GLWESwitchingKeyDegreesMut for LWEToGLWEKeyPrepared<D, B> {
    fn input_degree(&mut self) -> &mut Degree {
        &mut self.0.input_degree
    }

    fn output_degree(&mut self) -> &mut Degree {
        &mut self.0.output_degree
    }
}
