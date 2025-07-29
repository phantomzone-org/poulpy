use crate::{
    Module, NTT120, VecZnxBig, VecZnxBigBytesOf, ZnxInfos, ZnxSliceSize, ZnxView,
    vec_znx_big::impl_traits::VecZnxBigAllocBytesImpl,
};

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

impl VecZnxBigAllocBytesImpl<NTT120> for Module<NTT120> {
    fn vec_znx_big_alloc_bytes_impl(module: &Module<NTT120>, cols: usize, size: usize) -> usize {
        VecZnxBig::<Vec<u8>, NTT120>::bytes_of(module.n(), cols, size)
    }
}
