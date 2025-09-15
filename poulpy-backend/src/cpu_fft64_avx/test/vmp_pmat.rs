use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    test_suite::vmp::{test_vmp_apply_dft_to_dft, test_vmp_apply_dft_to_dft_add},
};

use crate::{FFT64Avx, FFT64Ref};

#[test]
fn vmp_apply_dft_to_dft_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vmp_apply_dft_to_dft(12, &module_ref, &module_test);
}

#[test]
fn vmp_apply_dft_to_dft_add_fft64() {
    let module_test: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << 5);
    let module_ref: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << 5);
    test_vmp_apply_dft_to_dft_add(12, &module_ref, &module_test);
}
