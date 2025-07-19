use backend::{Backend, FFT64, MatZnxDftPrep, MatZnxDftPrepAlloc, MatZnxDftPrepOps, Module, Scratch};

use crate::{GGLWECiphertext, Infos};

pub struct GGSWCiphertextPrep<C, B: Backend> {
    pub(crate) data: MatZnxDftPrep<C, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl GGSWCiphertextPrep<Vec<u8>, FFT64> {
    pub fn alloc(module: &Module<FFT64>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
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
            data: module.new_mat_znx_dft_prep(rows, rank + 1, rank + 1, k.div_ceil(basek)),
            basek,
            k: k,
            digits,
        }
    }

    pub fn bytes_of(module: &Module<FFT64>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
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

        module.bytes_of_mat_znx_dft_prep(rows, rank + 1, rank + 1, size)
    }
}

impl<T, B: Backend> Infos for GGSWCiphertextPrep<T, B> {
    type Inner = MatZnxDftPrep<T, B>;

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

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GGSWCiphertextPrep<DataSelf, FFT64> {
    pub fn prepare<DataOther>(&mut self, module: &Module<FFT64>, other: &GGLWECiphertext<DataOther>, scratch: &mut Scratch)
    where
        DataOther: AsRef<[u8]>,
    {
        module.vmp_prepare(&mut self.data, &other.data, scratch);
        self.k = other.k;
        self.basek = other.basek;
        self.digits = other.digits;
    }
}
