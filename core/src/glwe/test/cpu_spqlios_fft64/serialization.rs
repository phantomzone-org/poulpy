use backend::implementation::cpu_spqlios::FFT64;

use crate::glwe::test::serialization::{test_glwe_seeded_serialization, test_glwe_serialization};

#[test]
fn serialization() {
    test_glwe_serialization::<FFT64>();
    test_glwe_seeded_serialization::<FFT64>();
}
