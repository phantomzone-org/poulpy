use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat},
};

use crate::layouts::{
    GGLWECiphertext, Infos,
    prepared::{Prepare, PrepareAlloc},
};

#[derive(PartialEq, Eq)]
pub struct GGLWECiphertextPrepared<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<B: Backend> GGLWECiphertextPrepared<Vec<u8>, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize, rank_out: usize) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
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
            data: module.vmp_pmat_alloc(rows, rank_in, rank_out + 1, size),
            basek,
            k,
            digits,
        }
    }

    #[allow(clippy::too_many_arguments)]
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

        module.vmp_pmat_alloc_bytes(rows, rank_in, rank_out + 1, rows)
    }
}

impl<D: Data, B: Backend> Infos for GGLWECiphertextPrepared<D, B> {
    type Inner = VmpPMat<D, B>;

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

impl<D: Data, B: Backend> GGLWECiphertextPrepared<D, B> {
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

impl<D: DataMut, DR: DataRef, B: Backend> Prepare<B, GGLWECiphertext<DR>> for GGLWECiphertextPrepared<D, B>
where
    Module<B>: VmpPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GGLWECiphertext<DR>, scratch: &mut Scratch<B>) {
        module.vmp_prepare(&mut self.data, &other.data, scratch);
        self.basek = other.basek;
        self.k = other.k;
        self.digits = other.digits;
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GGLWECiphertextPrepared<Vec<u8>, B>> for GGLWECiphertext<D>
where
    Module<B>: VmpPMatAlloc<B> + VmpPrepare<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GGLWECiphertextPrepared<Vec<u8>, B> {
        let mut atk_prepared: GGLWECiphertextPrepared<Vec<u8>, B> = GGLWECiphertextPrepared::alloc(
            module,
            self.basek(),
            self.k(),
            self.rows(),
            self.digits(),
            self.rank_in(),
            self.rank_out(),
        );
        atk_prepared.prepare(module, self, scratch);
        atk_prepared
    }
}
