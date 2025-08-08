use crate::{
    hal::{
        api::{ZnxInfos, ZnxSliceSize, ZnxView},
        layouts::{Data, DataRef, VecZnxBig, VecZnxBigBytesOf},
        oep::VecZnxBigAllocBytesImpl,
    },
    implementation::cpu_spqlios::module_ntt120::NTT120,
};

const VEC_ZNX_BIG_NTT120_WORDSIZE: usize = 4;

impl<D: DataRef> ZnxView for VecZnxBig<D, NTT120> {
    type Scalar = i128;
}

impl<D: Data> VecZnxBigBytesOf for VecZnxBig<D, NTT120> {
    fn bytes_of(n: usize, cols: usize, size: usize) -> usize {
        VEC_ZNX_BIG_NTT120_WORDSIZE * n * cols * size * size_of::<f64>()
    }
}

impl<D: Data> ZnxSliceSize for VecZnxBig<D, NTT120> {
    fn sl(&self) -> usize {
        VEC_ZNX_BIG_NTT120_WORDSIZE * self.n() * self.cols()
    }
}

unsafe impl VecZnxBigAllocBytesImpl<NTT120> for NTT120 {
    fn vec_znx_big_alloc_bytes_impl(n: usize, cols: usize, size: usize) -> usize {
        VecZnxBig::<Vec<u8>, NTT120>::bytes_of(n, cols, size)
    }
}
