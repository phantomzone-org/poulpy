use backend::{Backend, MatZnx, Module, Scratch, VmpPMat};

use crate::{GGLWEExecLayoutFamily, GLWECiphertext, GLWESwitchingKey, GLWESwitchingKeyExec, Infos};

pub struct GGLWEAutomorphismKey<D> {
    pub(crate) key: GLWESwitchingKey<D>,
    pub(crate) p: i64,
}

impl GGLWEAutomorphismKey<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
        GGLWEAutomorphismKey {
            key: GLWESwitchingKey::alloc(module, basek, k, rows, digits, rank, rank),
            p: 0,
        }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        GLWESwitchingKey::<Vec<u8>>::bytes_of(module, basek, k, rows, digits, rank, rank)
    }
}

impl<D> Infos for GGLWEAutomorphismKey<D> {
    type Inner = MatZnx<D>;

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

impl<D> GGLWEAutomorphismKey<D> {
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

impl<D: AsRef<[u8]>> GGLWEAutomorphismKey<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        self.key.at(row, col)
    }
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> GGLWEAutomorphismKey<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        self.key.at_mut(row, col)
    }
}

pub struct GLWEAutomorphismKeyExec<D, B: Backend> {
    pub(crate) key: GLWESwitchingKeyExec<D, B>,
    pub(crate) p: i64,
}

impl<B: Backend> GLWEAutomorphismKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GLWEAutomorphismKeyExec::<Vec<u8>, B> {
            key: GLWESwitchingKeyExec::alloc(module, basek, k, rows, digits, rank, rank),
            p: 0,
        }
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GLWESwitchingKeyExec::<Vec<u8>, B>::bytes_of(module, basek, k, rows, digits, rank, rank)
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>, B: Backend> GLWEAutomorphismKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GGLWEAutomorphismKey<DataOther>, scratch: &mut Scratch)
    where
        DataOther: AsRef<[u8]>,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        self.key.prepare(module, &other.key, scratch);
        self.p = other.p;
    }
}

impl<D, B: Backend> Infos for GLWEAutomorphismKeyExec<D, B> {
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

impl<D, B: Backend> GLWEAutomorphismKeyExec<D, B> {
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
