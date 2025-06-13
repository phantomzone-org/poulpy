use backend::{Backend, FFT64, MatZnxDft, MatZnxDftOps, Module};

use crate::{FourierGLWECiphertext, GLWESwitchingKey, GetRow, Infos, SetRow};

pub struct GLWEAutomorphismKey<Data, B: Backend> {
    pub(crate) key: GLWESwitchingKey<Data, B>,
    pub(crate) p: i64,
}

impl GLWEAutomorphismKey<Vec<u8>, FFT64> {
    pub fn alloc(module: &Module<FFT64>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
        GLWEAutomorphismKey {
            key: GLWESwitchingKey::alloc(module, basek, k, rows, digits, rank, rank),
            p: 0,
        }
    }

    pub fn bytes_of(module: &Module<FFT64>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        GLWESwitchingKey::<Vec<u8>, FFT64>::bytes_of(module, basek, k, rows, digits, rank, rank)
    }
}

impl<T, B: Backend> Infos for GLWEAutomorphismKey<T, B> {
    type Inner = MatZnxDft<T, B>;

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

impl<T, B: Backend> GLWEAutomorphismKey<T, B> {
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

impl<C: AsRef<[u8]>> GetRow<FFT64> for GLWEAutomorphismKey<C, FFT64> {
    fn get_row<R: AsMut<[u8]> + AsRef<[u8]>>(
        &self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        res: &mut FourierGLWECiphertext<R, FFT64>,
    ) {
        module.mat_znx_dft_get_row(&mut res.data, &self.key.0.data, row_i, col_j);
    }
}

impl<C: AsMut<[u8]> + AsRef<[u8]>> SetRow<FFT64> for GLWEAutomorphismKey<C, FFT64> {
    fn set_row<R: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        a: &FourierGLWECiphertext<R, FFT64>,
    ) {
        module.mat_znx_dft_set_row(&mut self.key.0.data, row_i, col_j, &a.data);
    }
}
