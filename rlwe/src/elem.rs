use base2k::{Backend, FFT64, MatZnxDft, MatZnxDftAlloc, Module, VecZnx, VecZnxAlloc};

pub struct Ciphertext<T> {
    data: T,
    log_base2k: usize,
    log_q: usize,
}

impl<T> Ciphertext<T> {
    pub fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    pub fn log_q(&self) -> usize {
        self.log_q
    }

    pub fn data(&self) -> &T {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

pub struct Plaintext<T> {
    data: T,
    log_base2k: usize,
    log_q: usize,
}

impl<T> Plaintext<T> {
    pub fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    pub fn log_q(&self) -> usize {
        self.log_q
    }

    pub fn data(&self) -> &T {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

pub(crate) type CipherVecZnx<C> = Ciphertext<VecZnx<C>>;

impl Ciphertext<VecZnx<Vec<u8>>> {
    pub fn new<B: Backend>(module: &Module<B>, log_base2k: usize, log_q: usize, cols: usize) -> Self {
        Self {
            data: module.new_vec_znx(cols, derive_size(log_base2k, log_q)),
            log_base2k: log_base2k,
            log_q: log_q,
        }
    }
}

impl Plaintext<VecZnx<Vec<u8>>> {
    pub fn new<B: Backend>(module: &Module<B>, log_base2k: usize, log_q: usize) -> Self {
        Self {
            data: module.new_vec_znx(1, derive_size(log_base2k, log_q)),
            log_base2k: log_base2k,
            log_q: log_q,
        }
    }
}

impl<B: Backend> Ciphertext<MatZnxDft<Vec<u8>, B>> {
    pub fn new(module: &Module<B>, log_base2k: usize, rows: usize, cols_in: usize, cols_out: usize, log_q: usize) -> Self {
        Self {
            data: module.new_mat_znx_dft(rows, cols_in, cols_out, derive_size(log_base2k, log_q)),
            log_base2k: log_base2k,
            log_q: log_q,
        }
    }
}

pub(crate) fn derive_size(log_base2k: usize, log_q: usize) -> usize {
    (log_q + log_base2k - 1) / log_base2k
}
