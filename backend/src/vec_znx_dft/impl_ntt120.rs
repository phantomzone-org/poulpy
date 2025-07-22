use crate::{
    Module, NTT120, VecZnxDft, VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftBytesOf, VecZnxDftOwned, ZnxInfos, ZnxSliceSize,
    ZnxView,
};

const VEC_ZNX_DFT_NTT120_WORDSIZE: usize = 4;

impl<D> ZnxSliceSize for VecZnxDft<D, NTT120> {
    fn sl(&self) -> usize {
        VEC_ZNX_DFT_NTT120_WORDSIZE * self.n() * self.cols()
    }
}

impl<D: AsRef<[u8]>> VecZnxDftBytesOf for VecZnxDft<D, NTT120> {
    fn bytes_of(n: usize, cols: usize, size: usize) -> usize {
        VEC_ZNX_DFT_NTT120_WORDSIZE * n * cols * size * size_of::<i64>()
    }
}

impl<D: AsRef<[u8]>> ZnxView for VecZnxDft<D, NTT120> {
    type Scalar = i64;
}

impl VecZnxDftAllocBytes for Module<NTT120> {
    fn vec_znx_dft_alloc_bytes(&self, cols: usize, size: usize) -> usize {
        VecZnxDft::<Vec<u8>, NTT120>::bytes_of(self.n(), cols, size)
    }
}

impl VecZnxDftAlloc<NTT120> for Module<NTT120> {
    fn vec_znx_dft_alloc(&self, cols: usize, size: usize) -> VecZnxDftOwned<NTT120> {
        VecZnxDftOwned::alloc(self.n(), cols, size)
    }
}
