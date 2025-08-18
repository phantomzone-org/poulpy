use poulpy_hal::{layouts::Backend, oep::VecZnxBigAllocBytesImpl};

use crate::cpu_spqlios::NTT120;

unsafe impl VecZnxBigAllocBytesImpl<NTT120> for NTT120 {
    fn vec_znx_big_alloc_bytes_impl(n: usize, cols: usize, size: usize) -> usize {
        NTT120::layout_big_word_count() * n * cols * size * size_of::<i128>()
    }
}
