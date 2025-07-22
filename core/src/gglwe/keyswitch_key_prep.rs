use backend::{Backend, Module, Scratch, VmpPMat, VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatPrepare};

use crate::{GGLWECiphertextPrep, GLWESwitchingKey, Infos};

pub struct GLWESwitchingKeyExec<D, B: Backend> {
    pub(crate) key: GGLWECiphertextPrep<D, B>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

impl<D: AsRef<[u8]> + AsMut<[u8]>, B: Backend> GLWESwitchingKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GLWESwitchingKey<DataOther>, scratch: &mut Scratch)
    where
        DataOther: AsRef<[u8]>,
        Module<B>: VmpPMatPrepare<B>,
    {
        self.key.prepare(module, &other.key, scratch);
        self.sk_in_n = other.sk_in_n;
        self.sk_out_n = other.sk_out_n;
    }
}

impl<B: Backend> GLWESwitchingKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize, rank_out: usize) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        GLWESwitchingKeyExec::<Vec<u8>, B> {
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
    ) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GGLWECiphertextPrep::bytes_of(module, basek, k, rows, digits, rank_in, rank_out)
    }
}

impl<D, B: Backend> Infos for GLWESwitchingKeyExec<D, B> {
    type Inner = VmpPMat<D, B>;

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

impl<D, B: Backend> GLWESwitchingKeyExec<D, B> {
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
