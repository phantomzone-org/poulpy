use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    AutomorphismKeyToRef, Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    prepared::{
        GLWESwitchingKeyPrepareTmpBytes, GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedToMut, GLWESwitchingKeyPreparedToRef,
    },
};

#[derive(PartialEq, Eq)]
pub struct AutomorphismKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GLWESwitchingKeyPrepared<D, B>,
    pub(crate) p: i64,
}

impl<D: Data, B: Backend> AutomorphismKeyPrepared<D, B> {
    pub fn p(&self) -> i64 {
        self.p
    }
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

pub trait SetP {
    fn set_p(&mut self, p: i64);
}

impl<D: Data, B: Backend> SetP for AutomorphismKeyPrepared<D, B> {
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

impl<B: Backend> AutomorphismKeyPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAlloc<B>,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKeyPrepared"
        );
        AutomorphismKeyPrepared::<Vec<u8>, B> {
            key: GLWESwitchingKeyPrepared::alloc(module, infos),
            p: 0,
        }
    }

    pub fn alloc_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        AutomorphismKeyPrepared {
            key: GLWESwitchingKeyPrepared::alloc_with(module, base2k, k, rank, rank, dnum, dsize),
            p: 0,
        }
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAllocBytes,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKeyPrepared"
        );
        GLWESwitchingKeyPrepared::alloc_bytes(module, infos)
    }

    pub fn alloc_bytes_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GLWESwitchingKeyPrepared::alloc_bytes_with(module, base2k, k, rank, rank, dnum, dsize)
    }
}

pub trait AutomorphismKeyPrepareTmpBytes {
    fn automorphism_key_prepare_tmp_bytes<A>(&self, infos: &A)
    where
        A: GGLWEInfos;
}

impl<B: Backend> AutomorphismKeyPrepareTmpBytes for Module<B>
where
    Module<B>: GLWESwitchingKeyPrepareTmpBytes,
{
    fn automorphism_key_prepare_tmp_bytes<A>(&self, infos: &A)
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
        module.automorphism_key_prepare_tmp_bytes(self);
    }
}

pub trait AutomorphismKeyPrepare<B: Backend> {
    fn automorphism_key_prepare<R, O>(&self, res: &R, other: &O, scratch: &Scratch<B>)
    where
        R: AutomorphismKeyPreparedToMut<B>,
        O: AutomorphismKeyToRef;
}

impl<B: Backend> AutomorphismKeyPrepare<B> for Module<B> {
    fn automorphism_key_prepare<R, O>(&self, res: &R, other: &O, scratch: &Scratch<B>)
    where
        R: AutomorphismKeyPreparedToMut<B>,
        O: AutomorphismKeyToRef,
    {
        self.key.prepare(self, &other.to_ref().key, scratch);
        self.p = other.p;
    }
}

pub trait AutomorphismKeyPrepareAlloc<B: Backend> {
    fn automorphism_key_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>) -> AutomorphismKeyPrepared<Vec<u8>, B>
    where
        O: AutomorphismKeyToRef;
}

impl<B: Backend> AutomorphismKeyPrepareAlloc<B> for Module<B>
where
    Module<B>: VmpPMatAlloc<B> + VmpPrepare<B>,
{
    fn automorphism_key_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>) -> AutomorphismKeyPrepared<Vec<u8>, B>
    where
        O: AutomorphismKeyToRef,
    {
        let mut atk_prepared: AutomorphismKeyPrepared<Vec<u8>, B> = AutomorphismKeyPrepared::alloc(self, &other.to_ref());
        atk_prepared.prepare(self, &other.to_ref(), scratch);
        atk_prepared
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
