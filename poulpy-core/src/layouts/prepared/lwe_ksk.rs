use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, LWESwitchingKeyToRef, Rank, TorusPrecision,
    prepared::{GLWESwitchingKeyPrepare, GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedToMut, GLWESwitchingKeyPreparedToRef},
};

#[derive(PartialEq, Eq)]
pub struct LWESwitchingKeyPrepared<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for LWESwitchingKeyPrepared<D, B> {
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
impl<D: Data, B: Backend> GLWEInfos for LWESwitchingKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for LWESwitchingKeyPrepared<D, B> {
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

impl<B: Backend> LWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAlloc<B>,
    {
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWESwitchingKey"
        );
        debug_assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWESwitchingKey"
        );
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for LWESwitchingKey"
        );
        Self(GLWESwitchingKeyPrepared::alloc_from_infos(module, infos))
    }

    pub fn alloc_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        Self(GLWESwitchingKeyPrepared::alloc(
            module,
            base2k,
            k,
            Rank(1),
            Rank(1),
            dnum,
            Dsize(1),
        ))
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAllocBytes,
    {
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWESwitchingKey"
        );
        debug_assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWESwitchingKey"
        );
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for LWESwitchingKey"
        );
        GLWESwitchingKeyPrepared::alloc_bytes_from_infos(module, infos)
    }

    pub fn alloc_bytes_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GLWESwitchingKeyPrepared::alloc_bytes(module, base2k, k, Rank(1), Rank(1), dnum, Dsize(1))
    }
}

pub trait LWESwitchingKeyPrepareTmpBytes {
    fn lwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A)
    where
        A: GGLWEInfos;
}

impl<B: Backend> LWESwitchingKeyPrepareTmpBytes for Module<B>
where
    Module<B>: LWESwitchingKeyPrepareTmpBytes,
{
    fn lwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A)
    where
        A: GGLWEInfos,
    {
        self.lwe_switching_key_prepare_tmp_bytes(infos);
    }
}

impl<B: Backend> LWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<A>(&self, module: &Module<B>, infos: &A)
    where
        A: GLWEInfos,
        Module<B>: LWESwitchingKeyPrepareTmpBytes,
    {
        module.glwe_secret_prepare_tmp_bytes(infos);
    }
}

pub trait LWESwitchingKeyPrepare<B: Backend> {
    fn lwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &Scratch<B>)
    where
        R: LWESwitchingKeyPreparedToMut<B>,
        O: LWESwitchingKeyToRef;
}

impl<B: Backend> LWESwitchingKeyPrepare<B> for Module<B>
where
    Module<B>: GLWESwitchingKeyPrepare<B>,
{
    fn lwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &Scratch<B>)
    where
        R: LWESwitchingKeyPreparedToMut<B>,
        O: LWESwitchingKeyToRef,
    {
        self.glwe_switching_prepare(&mut res.to_mut().0, other, scratch);
    }
}

impl<D: DataMut, B: Backend> LWESwitchingKeyPrepared<D, B> {
    fn prepare<O>(&mut self, module: &Module<B>, other: &O, scratch: &Scratch<B>)
    where
        O: LWESwitchingKeyToRef,
        Module<B>: LWESwitchingKeyPrepare<B>,
    {
        module.lwe_switching_key_prepare(self, other, scratch);
    }
}

pub trait LWESwitchingKeyPrepareAlloc<B: Backend> {
    fn lwe_switching_key_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: LWESwitchingKeyToRef;
}

impl<B: Backend> LWESwitchingKeyPrepareAlloc<B> for Module<B>
where
    Module<B>: LWESwitchingKeyPrepare<B>,
{
    fn lwe_switching_key_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: LWESwitchingKeyToRef,
    {
        let mut ct_prep: LWESwitchingKeyPrepared<Vec<u8>, B> = LWESwitchingKeyPrepared::alloc(self, &other.to_ref());
        self.lwe_switching_key_prepare(&mut ct_prep, other, scratch);
        ct_prep
    }
}

pub trait LWESwitchingKeyPreparedToRef<B: Backend> {
    fn to_ref(&self) -> LWESwitchingKeyPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> LWESwitchingKeyPreparedToRef<B> for LWESwitchingKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToRef<B>,
{
    fn to_ref(&self) -> LWESwitchingKeyPrepared<&[u8], B> {
        LWESwitchingKeyPrepared(self.0.to_ref())
    }
}

pub trait LWESwitchingKeyPreparedToMut<B: Backend> {
    fn to_mut(&mut self) -> LWESwitchingKeyPrepared<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> LWESwitchingKeyPreparedToMut<B> for LWESwitchingKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToMut<B>,
{
    fn to_mut(&mut self) -> LWESwitchingKeyPrepared<&mut [u8], B> {
        LWESwitchingKeyPrepared(self.0.to_mut())
    }
}
