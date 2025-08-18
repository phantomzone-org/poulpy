use poulpy_hal::{
    layouts::{Backend, SvpPPolOwned},
    oep::{SvpPPolAllocBytesImpl, SvpPPolAllocImpl, SvpPPolFromBytesImpl},
};

use crate::cpu_spqlios::NTT120;

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
        NTT120::layout_prep_word_count() * n * cols * size_of::<i64>()
    }
}
