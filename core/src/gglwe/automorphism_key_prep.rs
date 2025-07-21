use backend::{Backend, Module, VmpPMat, VmpPMatAlloc, VmpPMatAllocBytes};

use crate::{GLWESwitchingKeyPrep, Infos};

pub struct GLWEAutomorphismKeyPrep<D, B: Backend> {
    pub(crate) key: GLWESwitchingKeyPrep<D, B>,
    pub(crate) p: i64,
}

impl<B: Backend> GLWEAutomorphismKeyPrep<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self where Module<B>: VmpPMatAlloc<B>{
        GLWEAutomorphismKeyPrep::<Vec<u8>, B> {
            key: GLWESwitchingKeyPrep::alloc(module, basek, k, rows, digits, rank, rank),
            p: 0,
        }
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize where Module<B>: VmpPMatAllocBytes{
        GLWESwitchingKeyPrep::<Vec<u8>, B>::bytes_of(module, basek, k, rows, digits, rank, rank)
    }
}

impl<D, B: Backend> Infos for GLWEAutomorphismKeyPrep<D, B> {
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

impl<D, B: Backend> GLWEAutomorphismKeyPrep<D, B> {
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
