use poulpy_hal::tests::vmp_pmat::test_vmp_apply;

use crate::{cpu_ref, cpu_spqlios};

#[test]
fn vmp_apply_cpu_spqlios_fft64() {
    test_vmp_apply::<cpu_spqlios::FFT64>();
}

#[test]
fn vmp_apply_cpu_ref_fft64() {
    test_vmp_apply::<cpu_ref::FFT64>();
}
