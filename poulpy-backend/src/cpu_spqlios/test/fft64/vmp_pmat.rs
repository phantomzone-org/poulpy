use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    test_suite::vmp::{test_vmp_apply_dft_to_dft, test_vmp_apply_dft_to_dft_add},
};

use crate::{cpu_fft64_ref::FFT64Ref, cpu_spqlios::FFT64Spqlios};

#[test]
fn vmp_apply_dft_to_dft_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vmp_apply_dft_to_dft(12, &module_ref, &module_test);
}

#[test]
fn vmp_apply_dft_to_dft_add_fft64() {
    let module_test: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vmp_apply_dft_to_dft_add(12, &module_ref, &module_test);
}
