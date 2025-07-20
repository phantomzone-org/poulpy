use crate::{
    Module, NTT120, VecZnxDft, VecZnxDftAllocBytes, VecZnxDftBytesOf, VecZnxDftNew, VecZnxDftOwned, ZnxInfos, ZnxSliceSize,
    ZnxView, ZnxWordSize,
};

impl<D> ZnxSliceSize for VecZnxDft<D, NTT120> {
    fn sl(&self) -> usize {
        Self::ws() * self.n() * self.cols()
    }
}

impl<D> ZnxWordSize for VecZnxDft<D, NTT120> {
    fn ws() -> usize {
        4
    }
}

impl<D: AsRef<[u8]>> ZnxView for VecZnxDft<D, NTT120> {
    type Scalar = i64;
}

impl VecZnxDftAllocBytes<NTT120> for Module<NTT120> {
    fn vec_znx_dft_alloc_bytes(&self, cols: usize, size: usize) -> usize {
        VecZnxDft::<Vec<u8>, NTT120>::bytes_of(self.n(), cols, size)
    }
}

impl VecZnxDftNew<NTT120> for Module<NTT120> {
    fn new_vec_znx_dft(&self, cols: usize, size: usize) -> VecZnxDftOwned<NTT120> {
        VecZnxDftOwned::new(self.n(), cols, size)
    }
}
