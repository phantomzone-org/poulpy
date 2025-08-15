use backend::hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat},
};

use crate::layouts::{
    GGLWEAutomorphismKey, Infos,
    prepared::{GGLWESwitchingKeyPrepared, Prepare, PrepareAlloc},
};

#[derive(PartialEq, Eq)]
pub struct GGLWEAutomorphismKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GGLWESwitchingKeyPrepared<D, B>,
    pub(crate) p: i64,
}

impl<B: Backend> GGLWEAutomorphismKeyPrepared<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        GGLWEAutomorphismKeyPrepared::<Vec<u8>, B> {
            key: GGLWESwitchingKeyPrepared::alloc(module, n, basek, k, rows, digits, rank, rank),
            p: 0,
        }
    }

    pub fn bytes_of(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GGLWESwitchingKeyPrepared::bytes_of(module, n, basek, k, rows, digits, rank, rank)
    }
}

impl<D: Data, B: Backend> Infos for GGLWEAutomorphismKeyPrepared<D, B> {
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

impl<D: Data, B: Backend> GGLWEAutomorphismKeyPrepared<D, B> {
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

impl<D: DataMut, DR: DataRef, B: Backend> Prepare<B, GGLWEAutomorphismKey<DR>> for GGLWEAutomorphismKeyPrepared<D, B>
where
    Module<B>: VmpPMatPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GGLWEAutomorphismKey<DR>, scratch: &mut Scratch<B>) {
        self.key.prepare(module, &other.key, scratch);
        self.p = other.p;
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GGLWEAutomorphismKeyPrepared<Vec<u8>, B>> for GGLWEAutomorphismKey<D>
where
    Module<B>: VmpPMatAlloc<B> + VmpPMatPrepare<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GGLWEAutomorphismKeyPrepared<Vec<u8>, B> {
        let mut atk_prepared: GGLWEAutomorphismKeyPrepared<Vec<u8>, B> = GGLWEAutomorphismKeyPrepared::alloc(
            module,
            self.n(),
            self.basek(),
            self.k(),
            self.rows(),
            self.digits(),
            self.rank(),
        );
        atk_prepared.prepare(module, self, scratch);
        atk_prepared
    }
}
