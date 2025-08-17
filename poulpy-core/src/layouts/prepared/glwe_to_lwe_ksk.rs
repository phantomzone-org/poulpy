use poulpy_backend::hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat},
};

use crate::layouts::{
    GLWEToLWESwitchingKey, Infos,
    prepared::{GGLWESwitchingKeyPrepared, Prepare, PrepareAlloc},
};

#[derive(PartialEq, Eq)]
pub struct GLWEToLWESwitchingKeyPrepared<D: Data, B: Backend>(pub(crate) GGLWESwitchingKeyPrepared<D, B>);

impl<D: Data, B: Backend> Infos for GLWEToLWESwitchingKeyPrepared<D, B> {
    type Inner = VmpPMat<D, B>;

    fn inner(&self) -> &Self::Inner {
        self.0.inner()
    }

    fn basek(&self) -> usize {
        self.0.basek()
    }

    fn k(&self) -> usize {
        self.0.k()
    }
}

impl<D: Data, B: Backend> GLWEToLWESwitchingKeyPrepared<D, B> {
    pub fn digits(&self) -> usize {
        self.0.digits()
    }

    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    pub fn rank_in(&self) -> usize {
        self.0.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.0.rank_out()
    }
}

impl<B: Backend> GLWEToLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, rank_in: usize) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        Self(GGLWESwitchingKeyPrepared::alloc(
            module, n, basek, k, rows, 1, rank_in, 1,
        ))
    }

    pub fn bytes_of(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GGLWESwitchingKeyPrepared::<Vec<u8>, B>::bytes_of(module, n, basek, k, rows, digits, rank_in, 1)
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GLWEToLWESwitchingKeyPrepared<Vec<u8>, B>> for GLWEToLWESwitchingKey<D>
where
    Module<B>: VmpPrepare<B> + VmpPMatAlloc<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GLWEToLWESwitchingKeyPrepared<Vec<u8>, B> {
        let mut ksk_prepared: GLWEToLWESwitchingKeyPrepared<Vec<u8>, B> = GLWEToLWESwitchingKeyPrepared::alloc(
            module,
            self.0.n(),
            self.0.basek(),
            self.0.k(),
            self.0.rows(),
            self.0.rank_in(),
        );
        ksk_prepared.prepare(module, self, scratch);
        ksk_prepared
    }
}

impl<DM: DataMut, DR: DataRef, B: Backend> Prepare<B, GLWEToLWESwitchingKey<DR>> for GLWEToLWESwitchingKeyPrepared<DM, B>
where
    Module<B>: VmpPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GLWEToLWESwitchingKey<DR>, scratch: &mut Scratch<B>) {
        self.0.prepare(module, &other.0, scratch);
    }
}
