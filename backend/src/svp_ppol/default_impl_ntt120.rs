use crate::{
    Module, NTT120, SvpPPol, SvpPPolAlloc, SvpPPolAllocBytes, SvpPPolBytesOf, SvpPPolFromBytes, SvpPPolOwned, ZnxInfos,
    ZnxSliceSize, ZnxView,
};

const SVP_PPOL_NTT120_WORD_SIZE: usize = 4;

impl<D: AsRef<[u8]>> SvpPPolBytesOf for SvpPPol<D, NTT120> {
    fn bytes_of(n: usize, cols: usize) -> usize {
        SVP_PPOL_NTT120_WORD_SIZE * n * cols * size_of::<i64>()
    }
}

impl<D> ZnxSliceSize for SvpPPol<D, NTT120> {
    fn sl(&self) -> usize {
        SVP_PPOL_NTT120_WORD_SIZE * self.n()
    }
}

impl<D: AsRef<[u8]>> ZnxView for SvpPPol<D, NTT120> {
    type Scalar = i64;
}

impl SvpPPolFromBytes<NTT120> for Module<NTT120> {
    fn svp_ppol_from_bytes(&self, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<NTT120> {
        SvpPPolOwned::from_bytes(self.n(), cols, bytes)
    }
}

impl SvpPPolAlloc<NTT120> for Module<NTT120> {
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<NTT120> {
        SvpPPolOwned::alloc(self.n(), cols)
    }
}

impl SvpPPolAllocBytes for Module<NTT120> {
    fn svp_ppol_alloc_bytes(&self, cols: usize) -> usize {
        SvpPPol::<Vec<u8>, NTT120>::bytes_of(self.n(), cols)
    }
}
