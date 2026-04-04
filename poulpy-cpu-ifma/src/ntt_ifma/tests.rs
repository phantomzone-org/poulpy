#[cfg(test)]
mod ntt_ifma_tests {
    use poulpy_hal::cross_backend_test_suite;

    cross_backend_test_suite! {
        mod vec_znx,
        backend_ref =  poulpy_cpu_ref::FFT64Ref,
        backend_test = crate::NTTIfma,
        params = TestParams { size: 1<<8, base2k: 12 },
        tests = {
            test_vec_znx_add => poulpy_hal::test_suite::vec_znx::test_vec_znx_add,
            test_vec_znx_sub => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub,
            test_vec_znx_negate => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate,
            test_vec_znx_normalize => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize,
            test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
        }
    }

    cross_backend_test_suite! {
        mod vec_znx_dft,
        backend_ref =  poulpy_cpu_ref::FFT64Ref,
        backend_test = crate::NTTIfma,
        params = TestParams { size: 1<<8, base2k: 12 },
        tests = {
            test_vec_znx_dft_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add,
            test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_tmpa => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_tmpa,
        }
    }

    cross_backend_test_suite! {
        mod svp,
        backend_ref =  poulpy_cpu_ref::FFT64Ref,
        backend_test = crate::NTTIfma,
        params = TestParams { size: 1<<8, base2k: 12 },
        tests = {
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        }
    }

    cross_backend_test_suite! {
        mod vmp,
        backend_ref =  poulpy_cpu_ref::FFT64Ref,
        backend_test = crate::NTTIfma,
        params = TestParams { size: 1<<8, base2k: 12 },
        tests = {
            test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        }
    }

    cross_backend_test_suite! {
        mod vec_znx_big,
        backend_ref =  poulpy_cpu_ref::FFT64Ref,
        backend_test = crate::NTTIfma,
        params = TestParams { size: 1<<8, base2k: 12 },
        tests = {
            test_vec_znx_big_normalize => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_normalize,
        }
    }
}
