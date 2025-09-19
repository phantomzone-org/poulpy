use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat},
};

use crate::layouts::{
    GGLWECiphertext, GGLWEMetadata, Infos,
    prepared::{Prepare, PrepareAlloc},
};

#[derive(PartialEq, Eq)]
pub struct GGLWECiphertextPrepared<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B>,
    pub(crate) metadata: GGLWEMetadata,
}

impl<D: Data, B: Backend> GGLWECiphertextPrepared<D, B> {
    pub fn metadata(&self) -> GGLWEMetadata {
        self.metadata
    }
}

impl<B: Backend> GGLWECiphertextPrepared<Vec<u8>, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn alloc(module: &Module<B>, metadata: GGLWEMetadata) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        let k: usize = metadata.k;
        let digits: usize = metadata.digits;
        let basek: usize = metadata.basek;
        let rows: usize = metadata.rows;
        let rank_in: usize = metadata.rank_in;
        let rank_out: usize = metadata.rank_out;

        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid gglwe: ceil(k/basek): {size} <= digits: {digits}"
        );

        assert!(
            rows * digits <= size,
            "invalid gglwe: rows: {rows} * digits:{digits} > ceil(k/basek): {size}"
        );

        Self {
            data: module.vmp_pmat_alloc(rows, rank_in, rank_out + 1, size),
            metadata,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bytes_of(module: &Module<B>, metadata: GGLWEMetadata) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        let k: usize = metadata.k;
        let digits: usize = metadata.digits;
        let basek: usize = metadata.basek;
        let rows: usize = metadata.rows;
        let rank_in: usize = metadata.rank_in;
        let rank_out: usize = metadata.rank_out;

        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid gglwe: ceil(k/basek): {size} <= digits: {digits}"
        );

        assert!(
            rows * digits <= size,
            "invalid gglwe: rows: {rows} * digits:{digits} > ceil(k/basek): {size}"
        );

        module.vmp_pmat_alloc_bytes(rows, rank_in, rank_out + 1, rows)
    }
}

impl<D: Data, B: Backend> Infos for GGLWECiphertextPrepared<D, B> {
    type Inner = VmpPMat<D, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.metadata.basek
    }

    fn k(&self) -> usize {
        self.metadata.k
    }
}

impl<D: Data, B: Backend> GGLWECiphertextPrepared<D, B> {
    pub fn digits(&self) -> usize {
        self.metadata.digits
    }

    pub fn rank_in(&self) -> usize {
        self.metadata.rank_in
    }

    pub fn rank_out(&self) -> usize {
        self.metadata.rank_out
    }
}

impl<D: DataMut, DR: DataRef, B: Backend> Prepare<B, GGLWECiphertext<DR>> for GGLWECiphertextPrepared<D, B>
where
    Module<B>: VmpPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GGLWECiphertext<DR>, scratch: &mut Scratch<B>) {
        module.vmp_prepare(&mut self.data, &other.data, scratch);
        self.metadata = other.metadata()
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GGLWECiphertextPrepared<Vec<u8>, B>> for GGLWECiphertext<D>
where
    Module<B>: VmpPMatAlloc<B> + VmpPrepare<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GGLWECiphertextPrepared<Vec<u8>, B> {
        let mut atk_prepared: GGLWECiphertextPrepared<Vec<u8>, B> = GGLWECiphertextPrepared::alloc(module, self.metadata());
        atk_prepared.prepare(module, self, scratch);
        atk_prepared
    }
}
