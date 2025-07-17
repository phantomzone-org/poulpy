use backend::{Backend, MatZnx, Module};

use crate::{GLWECiphertext, GLWESwitchingKey, Infos};

pub struct GLWEAutomorphismKey<D> {
    pub(crate) key: GLWESwitchingKey<D>,
    pub(crate) p: i64,
}

impl GLWEAutomorphismKey<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
        GLWEAutomorphismKey {
            key: GLWESwitchingKey::alloc(module, basek, k, rows, digits, rank, rank),
            p: 0,
        }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        GLWESwitchingKey::<Vec<u8>>::bytes_of(module, basek, k, rows, digits, rank, rank)
    }
}

impl<D> Infos for GLWEAutomorphismKey<D> {
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

impl<D> GLWEAutomorphismKey<D> {
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

impl<D: AsRef<[u8]>> GLWEAutomorphismKey<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        self.key.at(row, col)
    }
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> GLWEAutomorphismKey<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        self.key.at_mut(row, col)
    }
}
