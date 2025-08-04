use crate::{
    hal::{
        api::{ZnxInfos, ZnxSliceSize, ZnxView},
        layouts::{SvpPPol, SvpPPolBytesOf, SvpPPolOwned},
        oep::{SvpPPolAllocBytesImpl, SvpPPolAllocImpl, SvpPPolFromBytesImpl},
    },
    implementation::cpu_avx::module_ntt120::NTT120,
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

unsafe impl SvpPPolFromBytesImpl<Self> for NTT120 {
    fn svp_ppol_from_bytes_impl(n: usize, cols: usize, bytes: Vec<u8>) -> SvpPPolOwned<NTT120> {
        SvpPPolOwned::from_bytes(n, cols, bytes)
    }
}

unsafe impl SvpPPolAllocImpl<Self> for NTT120 {
    fn svp_ppol_alloc_impl(n: usize, cols: usize) -> SvpPPolOwned<NTT120> {
        SvpPPolOwned::alloc(n, cols)
    }
}

unsafe impl SvpPPolAllocBytesImpl<Self> for NTT120 {
    fn svp_ppol_alloc_bytes_impl(n: usize, cols: usize) -> usize {
        SvpPPol::<Vec<u8>, Self>::bytes_of(n, cols)
    }
}
