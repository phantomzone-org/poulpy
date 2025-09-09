use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    reference::vec_znx_dft::{test_vec_znx_dft_add, test_vec_znx_dft_add_inplace},
};

use crate::cpu_spqlios::FFT64;

#[test]
fn test_vec_znx_big_add_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_dft_add(&module);
}

#[test]
fn test_vec_znx_big_add_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_vec_znx_dft_add_inplace(&module);
}
