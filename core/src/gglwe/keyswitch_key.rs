use backend::{Backend, FFT64, MatZnxDft, MatZnxDftOps, Module};

use crate::{FourierGLWECiphertext, GGLWECiphertext, GetRow, Infos, SetRow};

pub struct GLWESwitchingKey<Data, B: Backend> {
    pub(crate) key: GGLWECiphertext<Data, B>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

impl GLWESwitchingKey<Vec<u8>, FFT64> {
    pub fn alloc(
        module: &Module<FFT64>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> Self {
        GLWESwitchingKey {
            key: GGLWECiphertext::alloc(module, basek, k, rows, digits, rank_in, rank_out),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn bytes_of(
        module: &Module<FFT64>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize {
        GGLWECiphertext::<Vec<u8>, FFT64>::bytes_of(module, basek, k, rows, digits, rank_in, rank_out)
    }
}

impl<T, B: Backend> Infos for GLWESwitchingKey<T, B> {
    type Inner = MatZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        self.key.inner()
    }

    fn basek(&self) -> usize {
        self.key.basek()
    }

    fn k(&self) -> usize {
        self.key.k()
    }
}

impl<T, B: Backend> GLWESwitchingKey<T, B> {
    pub fn rank(&self) -> usize {
        self.key.data.cols_out() - 1
    }

    pub fn rank_in(&self) -> usize {
        self.key.data.cols_in()
    }

    pub fn rank_out(&self) -> usize {
        self.key.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.key.digits()
    }

    pub fn sk_degree_in(&self) -> usize {
        self.sk_in_n
    }

    pub fn sk_degree_out(&self) -> usize {
        self.sk_out_n
    }
}

impl<C: AsRef<[u8]>> GetRow<FFT64> for GLWESwitchingKey<C, FFT64> {
    fn get_row<R: AsMut<[u8]> + AsRef<[u8]>>(
        &self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        res: &mut FourierGLWECiphertext<R, FFT64>,
    ) {
        module.mat_znx_dft_get_row(&mut res.data, &self.key.data, row_i, col_j);
    }
}

impl<C: AsMut<[u8]> + AsRef<[u8]>> SetRow<FFT64> for GLWESwitchingKey<C, FFT64> {
    fn set_row<R: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        a: &FourierGLWECiphertext<R, FFT64>,
    ) {
        module.mat_znx_dft_set_row(&mut self.key.data, row_i, col_j, &a.data);
    }
}
