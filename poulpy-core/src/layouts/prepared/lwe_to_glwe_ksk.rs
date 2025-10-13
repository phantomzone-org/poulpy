use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, LWEToGLWESwitchingKeyToRef, Rank, TorusPrecision,
    prepared::{GLWESwitchingKeyPrepare, GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedToMut, GLWESwitchingKeyPreparedToRef},
};

/// A special [GLWESwitchingKey] required to for the conversion from [LWECiphertext] to [GLWECiphertext].
#[derive(PartialEq, Eq)]
pub struct LWEToGLWESwitchingKeyPrepared<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for LWEToGLWESwitchingKeyPrepared<D, B> {
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

impl<D: Data, B: Backend> GLWEInfos for LWEToGLWESwitchingKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for LWEToGLWESwitchingKeyPrepared<D, B> {
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

impl<B: Backend> LWEToGLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAlloc<B>,
    {
        debug_assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWEToGLWESwitchingKey"
        );
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWEToGLWESwitchingKey"
        );
        Self(GLWESwitchingKeyPrepared::alloc_from_infos(module, infos))
    }

    pub fn alloc_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        Self(GLWESwitchingKeyPrepared::alloc(
            module,
            base2k,
            k,
            Rank(1),
            rank_out,
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
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWEToGLWESwitchingKey"
        );
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWEToGLWESwitchingKey"
        );
        GLWESwitchingKeyPrepared::alloc_bytes_from_infos(module, infos)
    }

    pub fn alloc_bytes_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, dnum: Dnum, rank_out: Rank) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GLWESwitchingKeyPrepared::alloc_bytes(module, base2k, k, Rank(1), rank_out, dnum, Dsize(1))
    }
}

pub trait LWEToGLWESwitchingKeyPrepareTmpBytes {
    fn lwe_to_glwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A)
    where
        A: GGLWEInfos;
}

impl<B: Backend> LWEToGLWESwitchingKeyPrepareTmpBytes for Module<B>
where
    Module<B>: LWEToGLWESwitchingKeyPrepareTmpBytes,
{
    fn lwe_to_glwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A)
    where
        A: GGLWEInfos,
    {
        self.lwe_to_glwe_switching_key_prepare_tmp_bytes(infos);
    }
}

impl<B: Backend> LWEToGLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<A>(&self, module: &Module<B>, infos: &A)
    where
        A: GLWEInfos,
        Module<B>: LWEToGLWESwitchingKeyPrepareTmpBytes,
    {
        module.glwe_secret_prepare_tmp_bytes(infos);
    }
}

pub trait LWEToGLWESwitchingKeyPrepare<B: Backend> {
    fn lwe_to_glwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &Scratch<B>)
    where
        R: LWEToGLWESwitchingKeyPreparedToMut<B>,
        O: LWEToGLWESwitchingKeyToRef;
}

impl<B: Backend> LWEToGLWESwitchingKeyPrepare<B> for Module<B>
where
    Module<B>: GLWESwitchingKeyPrepare<B>,
{
    fn lwe_to_glwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &Scratch<B>)
    where
        R: LWEToGLWESwitchingKeyPreparedToMut<B>,
        O: LWEToGLWESwitchingKeyToRef,
    {
        self.glwe_switching_prepare(&mut res.to_mut().0, other, scratch);
    }
}

impl<D: DataMut, B: Backend> LWEToGLWESwitchingKeyPrepared<D, B> {
    fn prepare<O>(&mut self, module: &Module<B>, other: &O, scratch: &Scratch<B>)
    where
        O: LWEToGLWESwitchingKeyToRef,
        Module<B>: LWEToGLWESwitchingKeyPrepare<B>,
    {
        module.lwe_to_glwe_switching_key_prepare(self, other, scratch);
    }
}

pub trait LWEToGLWESwitchingKeyPrepareAlloc<B: Backend> {
    fn lwe_to_glwe_switching_key_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: LWEToGLWESwitchingKeyToRef;
}

impl<B: Backend> LWEToGLWESwitchingKeyPrepareAlloc<B> for Module<B>
where
    Module<B>: LWEToGLWESwitchingKeyPrepare<B>,
{
    fn lwe_to_glwe_switching_key_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: LWEToGLWESwitchingKeyToRef,
    {
        let mut ct_prep: LWEToGLWESwitchingKeyPrepared<Vec<u8>, B> = LWEToGLWESwitchingKeyPrepared::alloc(self, &other.to_ref());
        self.lwe_to_glwe_switching_key_prepare(&mut ct_prep, other, scratch);
        ct_prep
    }
}

pub trait LWEToGLWESwitchingKeyPreparedToRef<B: Backend> {
    fn to_ref(&self) -> LWEToGLWESwitchingKeyPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> LWEToGLWESwitchingKeyPreparedToRef<B> for LWEToGLWESwitchingKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToRef<B>,
{
    fn to_ref(&self) -> LWEToGLWESwitchingKeyPrepared<&[u8], B> {
        LWEToGLWESwitchingKeyPrepared(self.0.to_ref())
    }
}

pub trait LWEToGLWESwitchingKeyPreparedToMut<B: Backend> {
    fn to_mut(&mut self) -> LWEToGLWESwitchingKeyPrepared<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> LWEToGLWESwitchingKeyPreparedToMut<B> for LWEToGLWESwitchingKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToMut<B>,
{
    fn to_mut(&mut self) -> LWEToGLWESwitchingKeyPrepared<&mut [u8], B> {
        LWEToGLWESwitchingKeyPrepared(self.0.to_mut())
    }
}
