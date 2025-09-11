use poulpy_hal::reference::vmp::fft64::{test_vmp_apply_dft_to_dft, test_vmp_prepare};

use crate::cpu_spqlios;

#[test]
fn vmp_prepare_fft64() {
    test_vmp_prepare::<cpu_spqlios::FFT64>();
}

#[test]
fn vmp_apply_dft_to_dft_fft64() {
    test_vmp_apply_dft_to_dft::<cpu_spqlios::FFT64>();
}
