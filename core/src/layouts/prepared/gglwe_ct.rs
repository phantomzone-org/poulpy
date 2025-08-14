use backend::hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat},
};

use crate::layouts::{GGLWECiphertext, Infos};

#[derive(PartialEq, Eq)]
pub struct GGLWECiphertextExec<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<B: Backend> GGLWECiphertextExec<Vec<u8>, B> {
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
            data: module.vmp_pmat_alloc(n, rows, rank_in, rank_out + 1, size),
            basek: basek,
            k,
            digits,
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

        module.vmp_pmat_alloc_bytes(n, rows, rank_in, rank_out + 1, rows)
    }
}

impl<D: Data, B: Backend> Infos for GGLWECiphertextExec<D, B> {
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

impl<D: Data, B: Backend> GGLWECiphertextExec<D, B> {
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

impl<D: DataMut, B: Backend> GGLWECiphertextExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GGLWECiphertext<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: VmpPMatPrepare<B>,
    {
        module.vmp_prepare(&mut self.data, &other.data, scratch);
        self.basek = other.basek;
        self.k = other.k;
        self.digits = other.digits;
    }
}
