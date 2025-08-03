use crate::{
    NTT120, VecZnxDft, VecZnxDftAllocBytesImpl, VecZnxDftAllocImpl, VecZnxDftBytesOf, VecZnxDftOwned, ZnxInfos, ZnxSliceSize,
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

unsafe impl VecZnxDftAllocBytesImpl<NTT120> for NTT120 {
    fn vec_znx_dft_alloc_bytes_impl(n: usize, cols: usize, size: usize) -> usize {
        VecZnxDft::<Vec<u8>, NTT120>::bytes_of(n, cols, size)
    }
}

unsafe impl VecZnxDftAllocImpl<NTT120> for NTT120 {
    fn vec_znx_dft_alloc_impl(n: usize, cols: usize, size: usize) -> VecZnxDftOwned<NTT120> {
        VecZnxDftOwned::alloc(n, cols, size)
    }
}
