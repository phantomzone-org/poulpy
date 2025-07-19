use backend::{Backend, MatZnxDftPrep, Module};

use crate::{GGLWECiphertextPrep, Infos};

pub struct GLWESwitchingKeyPrep<D, B: Backend> {
    pub(crate) key: GGLWECiphertextPrep<D, B>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

impl<B: Backend> GLWESwitchingKeyPrep<Vec<u8>, B> {
    pub fn alloc(
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> Self {
        GLWESwitchingKeyPrep::<Vec<u8>, B> {
            key: GGLWECiphertextPrep::alloc(module, basek, k, rows, digits, rank_in, rank_out),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn bytes_of(
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize {
        GGLWECiphertextPrep::bytes_of(module, basek, k, rows, digits, rank_in, rank_out)
    }
}

impl<D, B: Backend> Infos for GLWESwitchingKeyPrep<D, B> {
    type Inner = MatZnxDftPrep<D, B>;

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

impl<D, B: Backend> GLWESwitchingKeyPrep<D, B> {
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
