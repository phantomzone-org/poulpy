use backend::{Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, Module};

use crate::{FourierGLWECiphertext, GetRow, Infos, SetRow};

pub struct GGLWECiphertext<C, B: Backend> {
    pub(crate) data: MatZnxDft<C, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<B: Backend> GGLWECiphertext<Vec<u8>, B> {
    pub fn alloc(
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> Self {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid gglwe: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        Self {
            data: module.new_mat_znx_dft(rows, rank_in, rank_out + 1, size),
            basek: basek,
            k,
            digits,
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
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid gglwe: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        module.bytes_of_mat_znx_dft(rows, rank_in, rank_out + 1, rows)
    }
}

impl<T, B: Backend> Infos for GGLWECiphertext<T, B> {
    type Inner = MatZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<T, B: Backend> GGLWECiphertext<T, B> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.digits
    }

    pub fn rank_in(&self) -> usize {
        self.data.cols_in()
    }

    pub fn rank_out(&self) -> usize {
        self.data.cols_out() - 1
    }
}

impl<C: AsRef<[u8]>> GetRow<FFT64> for GGLWECiphertext<C, FFT64> {
    fn get_row<R: AsMut<[u8]> + AsRef<[u8]>>(
        &self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        res: &mut FourierGLWECiphertext<R, FFT64>,
    ) {
        module.mat_znx_dft_get_row(&mut res.data, &self.data, row_i, col_j);
    }
}

impl<C: AsMut<[u8]> + AsRef<[u8]>> SetRow<FFT64> for GGLWECiphertext<C, FFT64> {
    fn set_row<R: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        a: &FourierGLWECiphertext<R, FFT64>,
    ) {
        module.mat_znx_dft_set_row(&mut self.data, row_i, col_j, &a.data);
    }
}
