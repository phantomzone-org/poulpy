use backend::hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat},
};

use crate::layouts::{GGLWESwitchingKey, Infos, prepared::GGLWECiphertextExec};

#[derive(PartialEq, Eq)]
pub struct GGLWESwitchingKeyExec<D: Data, B: Backend> {
    pub(crate) key: GGLWECiphertextExec<D, B>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

impl<B: Backend> GGLWESwitchingKeyExec<Vec<u8>, B> {
    pub fn alloc(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        GGLWESwitchingKeyExec::<Vec<u8>, B> {
            key: GGLWECiphertextExec::alloc(module, n, basek, k, rows, digits, rank_in, rank_out),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn bytes_of(
        module: &Module<B>,
        n: usize,
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
        GGLWECiphertextExec::bytes_of(module, n, basek, k, rows, digits, rank_in, rank_out)
    }

    pub fn from<DataOther: DataRef>(module: &Module<B>, other: &GGLWESwitchingKey<DataOther>, scratch: &mut Scratch<B>) -> Self
    where
        Module<B>: VmpPMatAlloc<B> + VmpPMatPrepare<B>,
    {
        let mut ksk_exec: GGLWESwitchingKeyExec<Vec<u8>, B> = Self::alloc(
            module,
            other.n(),
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

impl<D: Data, B: Backend> Infos for GGLWESwitchingKeyExec<D, B> {
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

impl<D: Data, B: Backend> GGLWESwitchingKeyExec<D, B> {
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

impl<D: DataMut, B: Backend> GGLWESwitchingKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GGLWESwitchingKey<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: VmpPMatPrepare<B>,
    {
        self.key.prepare(module, &other.key, scratch);
        self.sk_in_n = other.sk_in_n;
        self.sk_out_n = other.sk_out_n;
    }
}
