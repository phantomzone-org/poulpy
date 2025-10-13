use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, GLWESwitchingKey, GLWESwitchingKeyToRef, LWEInfos, Rank, TorusPrecision,
    prepared::{GGLWEPrepare, GGLWEPrepareTmpBytes, GGLWEPrepared, GGLWEPreparedToMut, GGLWEPreparedToRef},
};

#[derive(PartialEq, Eq)]
pub struct GLWESwitchingKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GGLWEPrepared<D, B>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

pub(crate) trait GLWESwitchingKeyPreparedSetMetaData {
    fn set_sk_in_n(&mut self, sk_in_n: usize);
    fn set_sk_out_n(&mut self, sk_out_n: usize);
}

impl<D: DataMut, B: Backend> GLWESwitchingKeyPreparedSetMetaData for GLWESwitchingKeyPrepared<D, B> {
    fn set_sk_in_n(&mut self, sk_in_n: usize) {
        self.sk_in_n = sk_in_n
    }

    fn set_sk_out_n(&mut self, sk_out_n: usize) {
        self.sk_out_n = self.sk_out_n
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

impl<B: Backend> GLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAlloc<B>,
    {
        debug_assert_eq!(module.n() as u32, infos.n(), "module.n() != infos.n()");
        GLWESwitchingKeyPrepared::<Vec<u8>, B> {
            key: GGLWEPrepared::alloc(module, infos),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn alloc_with(
        module: &Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        GLWESwitchingKeyPrepared::<Vec<u8>, B> {
            key: GGLWEPrepared::alloc_with(module, base2k, k, rank_in, rank_out, dnum, dsize),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAllocBytes,
    {
        debug_assert_eq!(module.n() as u32, infos.n(), "module.n() != infos.n()");
        GGLWEPrepared::alloc_bytes(module, infos)
    }

    pub fn alloc_bytes_with(
        module: &Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GGLWEPrepared::alloc_bytes_with(module, base2k, k, rank_in, rank_out, dnum, dsize)
    }
}

pub trait GLWESwitchingKeyPrepareTmpBytes {
    fn glwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;
}

impl<B: Backend> GLWESwitchingKeyPrepareTmpBytes for Module<B>
where
    Module<B>: GGLWEPrepareTmpBytes,
{
    fn glwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.gglwe_prepare_tmp_bytes(infos)
    }
}

impl<B: Backend> GLWESwitchingKeyPrepared<Vec<u8>, B>
where
    Module<B>: GGLWEPrepareTmpBytes,
{
    pub fn prepare_tmp_bytes(&self, module: &Module<B>) -> usize {
        module.gglwe_prepare_tmp_bytes(self)
    }
}

pub trait GLWESwitchingKeyPrepare<B: Backend> {
    fn glwe_switching_prepare<R, O>(&self, res: &R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GLWESwitchingKeyPreparedToMut<B> + GLWESwitchingKeyPreparedSetMetaData,
        O: GLWESwitchingKeyToRef;
}

impl<B: Backend> GLWESwitchingKeyPrepare<B> for Module<B>
where
    Module<B>: GGLWEPrepare<B>,
{
    fn glwe_switching_prepare<R, O>(&self, res: &R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GLWESwitchingKeyPreparedToMut<B> + GLWESwitchingKeyPreparedSetMetaData,
        O: GLWESwitchingKeyToRef,
    {
        self.gglwe_prepare(&res.to_mut(), other, scratch);
        res.set_sk_in_n(other.sk_in_n);
        res.set_sk_out_n(other.sk_out_n);
    }
}

impl<D: DataMut, B: Backend> GLWESwitchingKeyPrepared<D, B> {
    pub fn prepare<O>(&mut self, module: &Module<B>, other: &O, scratch: &mut Scratch<B>)
    where
        O: GLWESwitchingKeyToRef,
        Module<B>: GLWESwitchingKeyPrepare<B>,
    {
        module.glwe_switching_prepare(self, other, scratch);
    }
}

pub trait GLWESwitchingKeyPrepareAlloc<B: Backend> {
    fn glwe_switching_key_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: GLWESwitchingKeyToRef;
}

impl<B: Backend> GLWESwitchingKeyPrepareAlloc<B> for Module<B>
where
    Module<B>: GLWESwitchingKeyPrepare<B>,
{
    fn glwe_switching_key_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: GLWESwitchingKeyToRef,
    {
        let mut ct_prepared: GLWESwitchingKeyPrepared<Vec<u8>, B> = GLWESwitchingKeyPrepared::alloc(self, self);
        self.glwe_switching_prepare(&mut ct_prepared, other, scratch);
        ct_prepared
    }
}

impl<D: DataRef> GLWESwitchingKey<D> {
    pub fn prepare_alloc<B: Backend>(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GLWESwitchingKeyPrepared<Vec<u8>, B>
    where
        Module<B>: GLWESwitchingKeyPrepareAlloc<B>,
    {
        module.glwe_switching_key_prepare_alloc(self, scratch);
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
