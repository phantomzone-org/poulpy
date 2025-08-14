use backend::hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat},
};

use crate::layouts::{GGLWEAutomorphismKey, Infos, prepared::GGLWESwitchingKeyExec};

#[derive(PartialEq, Eq)]
pub struct GGLWEAutomorphismKeyExec<D: Data, B: Backend> {
    pub(crate) key: GGLWESwitchingKeyExec<D, B>,
    pub(crate) p: i64,
}

impl<B: Backend> GGLWEAutomorphismKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        GGLWEAutomorphismKeyExec::<Vec<u8>, B> {
            key: GGLWESwitchingKeyExec::alloc(module, n, basek, k, rows, digits, rank, rank),
            p: 0,
        }
    }

    pub fn bytes_of(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GGLWESwitchingKeyExec::bytes_of(module, n, basek, k, rows, digits, rank, rank)
    }

    pub fn from<DataOther: DataRef>(module: &Module<B>, other: &GGLWEAutomorphismKey<DataOther>, scratch: &mut Scratch<B>) -> Self
    where
        Module<B>: VmpPMatAlloc<B> + VmpPMatPrepare<B>,
    {
        let mut atk_exec: GGLWEAutomorphismKeyExec<Vec<u8>, B> = Self::alloc(
            module,
            other.n(),
            other.basek(),
            other.k(),
            other.rows(),
            other.digits(),
            other.rank(),
        );
        atk_exec.prepare(module, other, scratch);
        atk_exec
    }
}

impl<D: Data, B: Backend> Infos for GGLWEAutomorphismKeyExec<D, B> {
    type Inner = VmpPMat<D, B>;

    fn inner(&self) -> &Self::Inner {
        &self.key.inner()
    }

    fn basek(&self) -> usize {
        self.key.basek()
    }

    fn k(&self) -> usize {
        self.key.k()
    }
}

impl<D: Data, B: Backend> GGLWEAutomorphismKeyExec<D, B> {
    pub fn p(&self) -> i64 {
        self.p
    }

    pub fn digits(&self) -> usize {
        self.key.digits()
    }

    pub fn rank(&self) -> usize {
        self.key.rank()
    }

    pub fn rank_in(&self) -> usize {
        self.key.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.key.rank_out()
    }
}

impl<D: DataMut, B: Backend> GGLWEAutomorphismKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GGLWEAutomorphismKey<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: VmpPMatPrepare<B>,
    {
        self.key.prepare(module, &other.key, scratch);
        self.p = other.p;
    }
}