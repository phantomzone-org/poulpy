use backend::{Backend, MatZnx, MatZnxAlloc, MatZnxAllocBytes, Module, Scratch, VmpPMat};

use crate::{GGLWECiphertext, GGLWECiphertextExec, GGLWEExecLayoutFamily, GLWECiphertext, Infos};

pub struct GLWESwitchingKey<D> {
    pub(crate) key: GGLWECiphertext<D>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

impl GLWESwitchingKey<Vec<u8>> {
    pub fn alloc<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        GLWESwitchingKey {
            key: GGLWECiphertext::alloc(module, basek, k, rows, digits, rank_in, rank_out),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn bytes_of<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize
    where
        Module<B>: MatZnxAllocBytes,
    {
        GGLWECiphertext::<Vec<u8>>::bytes_of(module, basek, k, rows, digits, rank_in, rank_out)
    }
}

impl<D> Infos for GLWESwitchingKey<D> {
    type Inner = MatZnx<D>;

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

impl<D> GLWESwitchingKey<D> {
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

impl<D: AsRef<[u8]>> GLWESwitchingKey<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        self.key.at(row, col)
    }
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> GLWESwitchingKey<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        self.key.at_mut(row, col)
    }
}

pub struct GLWESwitchingKeyExec<D, B: Backend> {
    pub(crate) key: GGLWECiphertextExec<D, B>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

impl<B: Backend> GLWESwitchingKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize, rank_out: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GLWESwitchingKeyExec::<Vec<u8>, B> {
            key: GGLWECiphertextExec::alloc(module, basek, k, rows, digits, rank_in, rank_out),
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
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GGLWECiphertextExec::bytes_of(module, basek, k, rows, digits, rank_in, rank_out)
    }

    pub fn from<DataOther: AsRef<[u8]>>(module: &Module<B>, other: &GLWESwitchingKey<DataOther>, scratch: &mut Scratch) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let mut ksk_exec: GLWESwitchingKeyExec<Vec<u8>, B> = Self::alloc(
            module,
            other.basek(),
            other.k(),
            other.rows(),
            other.digits(),
            other.rank_in(),
            other.rank_out(),
        );
        ksk_exec.prepare(module, other, scratch);
        ksk_exec
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

impl<D: AsRef<[u8]> + AsMut<[u8]>, B: Backend> GLWESwitchingKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GLWESwitchingKey<DataOther>, scratch: &mut Scratch)
    where
        DataOther: AsRef<[u8]>,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        self.key.prepare(module, &other.key, scratch);
        self.sk_in_n = other.sk_in_n;
        self.sk_out_n = other.sk_out_n;
    }
}
