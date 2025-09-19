use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat},
};

use crate::layouts::{
    GGLWEMetadata, GGLWESwitchingKey, Infos,
    prepared::{GGLWECiphertextPrepared, Prepare, PrepareAlloc},
};

#[derive(PartialEq, Eq)]
pub struct GGLWESwitchingKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GGLWECiphertextPrepared<D, B>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

impl<D: Data, B: Backend> GGLWESwitchingKeyPrepared<D, B> {
    pub fn metadata(&self) -> GGLWEMetadata {
        self.key.metadata()
    }
}

impl<B: Backend> GGLWESwitchingKeyPrepared<Vec<u8>, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn alloc(module: &Module<B>, metadata: GGLWEMetadata) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        GGLWESwitchingKeyPrepared::<Vec<u8>, B> {
            key: GGLWECiphertextPrepared::alloc(module, metadata),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bytes_of(module: &Module<B>, metadata: GGLWEMetadata) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GGLWECiphertextPrepared::bytes_of(module, metadata)
    }
}

impl<D: Data, B: Backend> Infos for GGLWESwitchingKeyPrepared<D, B> {
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

impl<D: Data, B: Backend> GGLWESwitchingKeyPrepared<D, B> {
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

impl<D: DataMut, DR: DataRef, B: Backend> Prepare<B, GGLWESwitchingKey<DR>> for GGLWESwitchingKeyPrepared<D, B>
where
    Module<B>: VmpPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GGLWESwitchingKey<DR>, scratch: &mut Scratch<B>) {
        self.key.prepare(module, &other.key, scratch);
        self.sk_in_n = other.sk_in_n;
        self.sk_out_n = other.sk_out_n;
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GGLWESwitchingKeyPrepared<Vec<u8>, B>> for GGLWESwitchingKey<D>
where
    Module<B>: VmpPMatAlloc<B> + VmpPrepare<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GGLWESwitchingKeyPrepared<Vec<u8>, B> {
        let mut atk_prepared: GGLWESwitchingKeyPrepared<Vec<u8>, B> = GGLWESwitchingKeyPrepared::alloc(
            module,
            self.metadata(),
        );
        atk_prepared.prepare(module, self, scratch);
        atk_prepared
    }
}
