use backend::{Backend, FFT64, MatZnxDft, MatZnxDftOps, Module};

use crate::{FourierGLWECiphertext, GGLWECiphertext, GetRow, Infos, SetRow};

pub struct GLWESwitchingKey<Data, B: Backend>(pub(crate) GGLWECiphertext<Data, B>);

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
        GLWESwitchingKey(GGLWECiphertext::alloc(
            module, basek, k, rows, digits, rank_in, rank_out,
        ))
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
        self.0.inner()
    }

    fn basek(&self) -> usize {
        self.0.basek()
    }

    fn k(&self) -> usize {
        self.0.k()
    }
}

impl<T, B: Backend> GLWESwitchingKey<T, B> {
    pub fn rank(&self) -> usize {
        self.0.data.cols_out() - 1
    }

    pub fn rank_in(&self) -> usize {
        self.0.data.cols_in()
    }

    pub fn rank_out(&self) -> usize {
        self.0.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.0.digits()
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
        module.mat_znx_dft_get_row(&mut res.data, &self.0.data, row_i, col_j);
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
        module.mat_znx_dft_set_row(&mut self.0.data, row_i, col_j, &a.data);
    }
}
