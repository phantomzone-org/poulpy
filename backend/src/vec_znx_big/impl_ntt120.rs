use crate::{Module, NTT120, VecZnxBig, VecZnxBigAllocBytes, VecZnxBigBytesOf, ZnxInfos, ZnxSliceSize, ZnxView};

const VEC_ZNX_BIG_NTT120_WORDSIZE: usize = 4;

impl<D: AsRef<[u8]>> ZnxView for VecZnxBig<D, NTT120> {
    type Scalar = i128;
}

impl<D: AsRef<[u8]>> VecZnxBigBytesOf for VecZnxBig<D, NTT120> {
    fn bytes_of(n: usize, cols: usize, size: usize) -> usize {
        VEC_ZNX_BIG_NTT120_WORDSIZE * n * cols * size * size_of::<f64>()
    }
}

impl<D: AsRef<[u8]>> ZnxSliceSize for VecZnxBig<D, NTT120> {
    fn sl(&self) -> usize {
        VEC_ZNX_BIG_NTT120_WORDSIZE * self.n() * self.cols()
    }
}

impl VecZnxBigAllocBytes for Module<NTT120> {
    fn vec_znx_big_alloc_bytes(&self, cols: usize, size: usize) -> usize {
        VecZnxBig::<Vec<u8>, NTT120>::bytes_of(self.n(), cols, size)
    }
}
