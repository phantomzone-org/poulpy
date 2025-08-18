use poulpy_hal::tests::vmp_pmat::test_vmp_apply;

use crate::cpu_spqlios::FFT64;

#[test]
fn vmp_apply() {
    test_vmp_apply::<FFT64>();
}
