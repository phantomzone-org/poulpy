use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, GLWEToLWESwitchingKeyToMut, GLWEToLWESwitchingKeyToRef, LWEInfos, Rank,
    TorusPrecision,
    prepared::{
        GLWESecretPrepareTmpBytes, GLWESwitchingKeyPrepare, GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedToMut,
        GLWESwitchingKeyPreparedToRef,
    },
};

#[derive(PartialEq, Eq)]
pub struct GLWEToLWESwitchingKeyPrepared<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for GLWEToLWESwitchingKeyPrepared<D, B> {
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

impl<D: Data, B: Backend> GLWEInfos for GLWEToLWESwitchingKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GLWEToLWESwitchingKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

impl<B: Backend> GLWEToLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAlloc<B>,
    {
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for GLWEToLWESwitchingKeyPrepared"
        );
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for GLWEToLWESwitchingKeyPrepared"
        );
        Self(GLWESwitchingKeyPrepared::alloc_from_infos(module, infos))
    }

    pub fn alloc_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        Self(GLWESwitchingKeyPrepared::alloc(
            module,
            base2k,
            k,
            rank_in,
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
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for GLWEToLWESwitchingKeyPrepared"
        );
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for GLWEToLWESwitchingKeyPrepared"
        );
        GLWESwitchingKeyPrepared::alloc_bytes_from_infos(module, infos)
    }

    pub fn alloc_bytes_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GLWESwitchingKeyPrepared::alloc_bytes(module, base2k, k, rank_in, Rank(1), dnum, Dsize(1))
    }
}

pub trait GLWEToLWESwitchingKeyPrepareTmpBytes {
    fn glwe_to_lwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A)
    where
        A: GGLWEInfos;
}

impl<B: Backend> GLWEToLWESwitchingKeyPrepareTmpBytes for Module<B>
where
    Module<B>: GLWEToLWESwitchingKeyPrepareTmpBytes,
{
    fn glwe_to_lwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A)
    where
        A: GGLWEInfos,
    {
        self.glwe_to_lwe_switching_key_prepare_tmp_bytes(infos);
    }
}

impl<B: Backend> GLWEToLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn prepare_tmp_bytes<A>(&self, module: &Module<B>, infos: &A)
    where
        A: GLWEInfos,
        Module<B>: GLWEToLWESwitchingKeyPrepareTmpBytes,
    {
        module.glwe_secret_prepare_tmp_bytes(infos);
    }
}

pub trait GLWEToLWESwitchingKeyPrepare<B: Backend> {
    fn glwe_to_lwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &Scratch<B>)
    where
        R: GLWEToLWESwitchingKeyPreparedToMut<B>,
        O: GLWEToLWESwitchingKeyToRef;
}

impl<B: Backend> GLWEToLWESwitchingKeyPrepare<B> for Module<B>
where
    Module<B>: GLWESwitchingKeyPrepare<B>,
{
    fn glwe_to_lwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &Scratch<B>)
    where
        R: GLWEToLWESwitchingKeyPreparedToMut<B>,
        O: GLWEToLWESwitchingKeyToRef,
    {
        self.glwe_switching_prepare(&mut res.to_mut().0, other, scratch);
    }
}

impl<D: DataMut, B: Backend> GLWEToLWESwitchingKeyPrepared<D, B> {
    fn prepare<O>(&mut self, module: &Module<B>, other: &O, scratch: &Scratch<B>)
    where
        O: GLWEToLWESwitchingKeyToRef,
        Module<B>: GLWEToLWESwitchingKeyPrepare<B>,
    {
        module.glwe_to_lwe_switching_key_prepare(self, other, scratch);
    }
}

pub trait GLWEToLWESwitchingKeyPrepareAlloc<B: Backend> {
    fn glwe_to_lwe_switching_key_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: GLWEToLWESwitchingKeyToRef;
}

impl<B: Backend> GLWEToLWESwitchingKeyPrepareAlloc<B> for Module<B>
where
    Module<B>: GLWEToLWESwitchingKeyPrepare<B>,
{
    fn glwe_to_lwe_switching_key_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: GLWEToLWESwitchingKeyToRef,
    {
        let mut ct_prep: GLWEToLWESwitchingKeyPrepared<Vec<u8>, B> = GLWEToLWESwitchingKeyPrepared::alloc(self, &other.to_ref());
        self.glwe_to_lwe_switching_key_prepare(&mut ct_prep, other, scratch);
        ct_prep
    }
}

pub trait GLWEToLWESwitchingKeyPreparedToRef<B: Backend> {
    fn to_ref(&self) -> GLWEToLWESwitchingKeyPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> GLWEToLWESwitchingKeyPreparedToRef<B> for GLWEToLWESwitchingKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToRef<B>,
{
    fn to_ref(&self) -> GLWEToLWESwitchingKeyPrepared<&[u8], B> {
        GLWEToLWESwitchingKeyPrepared(self.0.to_ref())
    }
}

pub trait GLWEToLWESwitchingKeyPreparedToMut<B: Backend> {
    fn to_mut(&mut self) -> GLWEToLWESwitchingKeyPrepared<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> GLWEToLWESwitchingKeyPreparedToMut<B> for GLWEToLWESwitchingKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToMut<B>,
{
    fn to_mut(&mut self) -> GLWEToLWESwitchingKeyPrepared<&mut [u8], B> {
        GLWEToLWESwitchingKeyPrepared(self.0.to_mut())
    }
}
