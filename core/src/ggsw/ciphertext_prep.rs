use backend::{Backend, Module, Scratch, VmpPMat, VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatPrepare};

use crate::{GGLWECiphertext, Infos};

pub struct GGSWCiphertextPrep<C, B: Backend> {
    pub(crate) data: VmpPMat<C, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<B: Backend> GGSWCiphertextPrep<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self where Module<B>: VmpPMatAlloc<B>{
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
            data: module.vmp_pmat_alloc(rows, rank + 1, rank + 1, k.div_ceil(basek)),
            basek,
            k: k,
            digits,
        }
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize where Module<B>: VmpPMatAllocBytes{
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

        module.vmp_pmat_alloc_bytes(rows, rank + 1, rank + 1, size)
    }
}

impl<T, B: Backend> Infos for GGSWCiphertextPrep<T, B> {
    type Inner = VmpPMat<T, B>;

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

impl<T, B: Backend> GGSWCiphertextPrep<T, B> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.digits
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>, B: Backend> GGSWCiphertextPrep<DataSelf, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GGLWECiphertext<DataOther>, scratch: &mut Scratch)
    where
        DataOther: AsRef<[u8]>,
        Module<B>: VmpPMatPrepare<B>
    {
        module.vmp_prepare(&mut self.data, &other.data, scratch);
        self.k = other.k;
        self.basek = other.basek;
        self.digits = other.digits;
    }
}
