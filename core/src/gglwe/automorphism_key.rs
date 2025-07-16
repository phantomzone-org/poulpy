use backend::{Backend, MatZnx, Module, Scratch, VmpPMat};

use crate::{GGLWEExecLayoutFamily, GLWECiphertext, GLWESwitchingKey, GLWESwitchingKeyExec, Infos};

pub struct AutomorphismKey<D> {
    pub(crate) key: GLWESwitchingKey<D>,
    pub(crate) p: i64,
}

impl AutomorphismKey<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
        AutomorphismKey {
            key: GLWESwitchingKey::alloc(module, basek, k, rows, digits, rank, rank),
            p: 0,
        }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        GLWESwitchingKey::<Vec<u8>>::bytes_of(module, basek, k, rows, digits, rank, rank)
    }
}

impl<D> Infos for AutomorphismKey<D> {
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

impl<D> AutomorphismKey<D> {
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

impl<D: AsRef<[u8]>> AutomorphismKey<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        self.key.at(row, col)
    }
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> AutomorphismKey<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        self.key.at_mut(row, col)
    }
}

pub struct AutomorphismKeyExec<D, B: Backend> {
    pub(crate) key: GLWESwitchingKeyExec<D, B>,
    pub(crate) p: i64,
}

impl<B: Backend> AutomorphismKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        AutomorphismKeyExec::<Vec<u8>, B> {
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

    pub fn from<DataOther: AsRef<[u8]>>(module: &Module<B>, other: &AutomorphismKey<DataOther>, scratch: &mut Scratch) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let mut atk_exec: AutomorphismKeyExec<Vec<u8>, B> = Self::alloc(
            module,
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

impl<D: AsRef<[u8]> + AsMut<[u8]>, B: Backend> AutomorphismKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &AutomorphismKey<DataOther>, scratch: &mut Scratch)
    where
        DataOther: AsRef<[u8]>,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        self.key.prepare(module, &other.key, scratch);
        self.p = other.p;
    }
}

impl<D, B: Backend> Infos for AutomorphismKeyExec<D, B> {
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

impl<D, B: Backend> AutomorphismKeyExec<D, B> {
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
