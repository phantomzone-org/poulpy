use poulpy_hal::{
    layouts::{Backend, VecZnxDftOwned},
    oep::{VecZnxDftAllocBytesImpl, VecZnxDftAllocImpl},
};

use crate::cpu_spqlios::NTT120;

unsafe impl VecZnxDftAllocBytesImpl<NTT120> for NTT120 {
    fn vec_znx_dft_alloc_bytes_impl(n: usize, cols: usize, size: usize) -> usize {
        NTT120::layout_prep_word_count() * n * cols * size * size_of::<i64>()
    }
}

unsafe impl VecZnxDftAllocImpl<NTT120> for NTT120 {
    fn vec_znx_dft_alloc_impl(n: usize, cols: usize, size: usize) -> VecZnxDftOwned<NTT120> {
        VecZnxDftOwned::alloc(n, cols, size)
    }
}
