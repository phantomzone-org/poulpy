use backend::hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat},
};

use crate::layouts::{GGSWCiphertext, Infos};

#[derive(PartialEq, Eq)]
pub struct GGSWCiphertextExec<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<B: Backend> GGSWCiphertextExec<Vec<u8>, B> {
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

    pub fn from<DataOther: DataRef>(
        module: &Module<B>,
        other: &GGSWCiphertext<DataOther>,
        scratch: &mut Scratch<B>,
    ) -> GGSWCiphertextExec<Vec<u8>, B>
    where
        Module<B>: VmpPMatAlloc<B> + VmpPMatPrepare<B>,
    {
        let mut ggsw_exec: GGSWCiphertextExec<Vec<u8>, B> = Self::alloc(
            module,
            other.n(),
            other.basek(),
            other.k(),
            other.rows(),
            other.digits(),
            other.rank(),
        );
        ggsw_exec.prepare(module, other, scratch);
        ggsw_exec
    }
}

impl<D: Data, B: Backend> Infos for GGSWCiphertextExec<D, B> {
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

impl<D: Data, B: Backend> GGSWCiphertextExec<D, B> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.digits
    }
}

impl<DataSelf: DataMut, B: Backend> GGSWCiphertextExec<DataSelf, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GGSWCiphertext<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: VmpPMatPrepare<B>,
    {
        module.vmp_prepare(&mut self.data, &other.data, scratch);
        self.k = other.k;
        self.basek = other.basek;
        self.digits = other.digits;
    }
}
