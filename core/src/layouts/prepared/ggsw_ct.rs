use backend::hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat},
};

use crate::layouts::{
    GGSWCiphertext, Infos,
    prepared::{Prepare, PrepareAlloc},
};

#[derive(PartialEq, Eq)]
pub struct GGSWCiphertextPrepared<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<B: Backend> GGSWCiphertextPrepared<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        let size: usize = k.div_ceil(basek);
        debug_assert!(digits > 0, "invalid ggsw: `digits` == 0");

        debug_assert!(
            size > digits,
            "invalid ggsw: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid ggsw: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        Self {
            data: module.vmp_pmat_alloc(n, rows, rank + 1, rank + 1, k.div_ceil(basek)),
            basek,
            k: k,
            digits,
        }
    }

    pub fn bytes_of(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid ggsw: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid ggsw: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        module.vmp_pmat_alloc_bytes(n, rows, rank + 1, rank + 1, size)
    }
}

impl<D: Data, B: Backend> Infos for GGSWCiphertextPrepared<D, B> {
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

impl<D: Data, B: Backend> GGSWCiphertextPrepared<D, B> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.digits
    }
}

impl<D: DataRef, B: Backend> GGSWCiphertextPrepared<D, B> {
    pub fn data(&self) -> &VmpPMat<D, B> {
        &self.data
    }
}

impl<D: DataMut, DR: DataRef, B: Backend> Prepare<B, GGSWCiphertext<DR>> for GGSWCiphertextPrepared<D, B>
where
    Module<B>: VmpPMatPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GGSWCiphertext<DR>, scratch: &mut Scratch<B>) {
        module.vmp_prepare(&mut self.data, &other.data, scratch);
        self.k = other.k;
        self.basek = other.basek;
        self.digits = other.digits;
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GGSWCiphertextPrepared<Vec<u8>, B>> for GGSWCiphertext<D>
where
    Module<B>: VmpPMatAlloc<B> + VmpPMatPrepare<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GGSWCiphertextPrepared<Vec<u8>, B> {
        let mut ggsw_prepared: GGSWCiphertextPrepared<Vec<u8>, B> = GGSWCiphertextPrepared::alloc(
            module,
            self.n(),
            self.basek(),
            self.k(),
            self.rows(),
            self.digits(),
            self.rank(),
        );
        ggsw_prepared.prepare(module, self, scratch);
        ggsw_prepared
    }
}
