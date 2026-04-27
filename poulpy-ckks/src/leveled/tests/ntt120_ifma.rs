use crate::ckks_backend_test_suite;

const ATK_ROTATIONS: &[i64] = &[1, 7];

ckks_backend_test_suite!(
    mod f64_tests,
    backend = poulpy_cpu_avx512::NTT120Ifma,
    scalar = f64,
    params = crate::leveled::tests::test_suite::NTT120_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod f128_tests,
    backend = poulpy_cpu_avx512::NTT120Ifma,
    scalar = f128::f128,
    params = crate::leveled::tests::test_suite::NTT120_PARAMS_F128,
    rotations = super::ATK_ROTATIONS,
);
