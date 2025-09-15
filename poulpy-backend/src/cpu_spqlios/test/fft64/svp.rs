use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    test_suite::svp::{test_svp_apply_dft_to_dft, test_svp_apply_dft_to_dft_inplace},
};

use crate::{cpu_fft64_ref::FFT64Ref, cpu_spqlios::FFT64Spqlios};

#[test]
fn test_svp_apply_dft_to_dft_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_svp_apply_dft_to_dft(12, &module_ref, &module_test);
}

#[test]
fn test_svp_apply_dft_to_dft_inplace_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_svp_apply_dft_to_dft_inplace(12, &module_ref, &module_test);
}
