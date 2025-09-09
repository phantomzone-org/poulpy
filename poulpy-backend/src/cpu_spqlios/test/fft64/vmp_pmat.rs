use poulpy_hal::reference::vmp::fft64::test_vmp_apply_dft_to_dft;

use crate::cpu_spqlios;

#[test]
fn vmp_apply_dft_to_dftfft64() {
    test_vmp_apply_dft_to_dft::<cpu_spqlios::FFT64>();
}
