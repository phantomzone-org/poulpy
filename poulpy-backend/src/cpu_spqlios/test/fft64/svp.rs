use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    reference::svp::fft64::{test_svp_apply_dft_to_dft, test_svp_apply_dft_to_dft_inplace, test_svp_prepare},
};

use crate::cpu_spqlios::FFT64;

#[test]
fn test_svp_prepare_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_svp_prepare(&module);
}

#[test]
fn test_svp_apply_dft_to_dft_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_svp_apply_dft_to_dft(&module);
}

#[test]
fn test_svp_apply_dft_to_dft_inplace_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 5);
    test_svp_apply_dft_to_dft_inplace(&module);
}
