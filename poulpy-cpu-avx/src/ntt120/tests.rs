use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    test_suite::convolution::{test_convolution, test_convolution_by_const, test_convolution_pairwise},
};

use crate::NTT120Avx;

#[cfg(test)]
mod poulpy_cpu_avx {
    use poulpy_hal::{backend_test_suite, cross_backend_test_suite};

    cross_backend_test_suite! {
        mod vec_znx,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT120Avx,
        size = 1<<8,
        base2k = 50,
        tests = {
            test_vec_znx_add => poulpy_hal::test_suite::vec_znx::test_vec_znx_add,
            test_vec_znx_add_inplace => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_inplace,
            test_vec_znx_add_scalar => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_scalar,
            test_vec_znx_add_scalar_inplace => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_scalar_inplace,
            test_vec_znx_sub => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub,
            test_vec_znx_sub_inplace => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_inplace,
            test_vec_znx_sub_negate_inplace => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_negate_inplace,
            test_vec_znx_sub_scalar => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_scalar,
            test_vec_znx_sub_scalar_inplace => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_scalar_inplace,
            test_vec_znx_rsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh,
            test_vec_znx_rsh_inplace => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_inplace,
            test_vec_znx_lsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh,
            test_vec_znx_lsh_inplace => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_inplace,
            test_vec_znx_negate => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate,
            test_vec_znx_negate_inplace => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_inplace,
            test_vec_znx_rotate => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate,
            test_vec_znx_rotate_inplace => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate_inplace,
            test_vec_znx_automorphism => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism,
            test_vec_znx_automorphism_inplace => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism_inplace,
            test_vec_znx_mul_xp_minus_one => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one,
            test_vec_znx_mul_xp_minus_one_inplace => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one_inplace,
            test_vec_znx_normalize => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize,
            test_vec_znx_normalize_inplace => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_inplace,
            test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
            test_vec_znx_split_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_split_ring,
            test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
        }
    }
    cross_backend_test_suite! {
        mod svp,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT120Avx,
        size = 1<<8,
        base2k = 50,
        tests = {
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
            test_svp_apply_dft_to_dft_inplace => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_inplace,
        }
    }
    cross_backend_test_suite! {
        mod vec_znx_big,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT120Avx,
        size = 1<<8,
        base2k = 50,
        tests = {
            test_vec_znx_big_add => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add,
            test_vec_znx_big_add_inplace => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_inplace,
            test_vec_znx_big_add_small => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small,
            test_vec_znx_big_add_small_inplace => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small_inplace,
            test_vec_znx_big_sub => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub,
            test_vec_znx_big_sub_inplace => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_inplace,
            test_vec_znx_big_automorphism => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_automorphism,
            test_vec_znx_big_automorphism_inplace => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_automorphism_inplace,
            test_vec_znx_big_negate => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_negate,
            test_vec_znx_big_negate_inplace => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_negate_inplace,
            test_vec_znx_big_normalize => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_normalize,
            test_vec_znx_big_sub_negate_inplace => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_negate_inplace,
            test_vec_znx_big_sub_small_a => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a,
            test_vec_znx_big_sub_small_a_inplace => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a_inplace,
            test_vec_znx_big_sub_small_b => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b,
            test_vec_znx_big_sub_small_b_inplace => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b_inplace,
        }
    }
    cross_backend_test_suite! {
        mod vec_znx_dft,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT120Avx,
        size = 1<<8,
        base2k = 50,
        tests = {
            test_vec_znx_dft_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add,
            test_vec_znx_dft_add_inplace => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_inplace,
            test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
            test_vec_znx_dft_sub_inplace => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_inplace,
            test_vec_znx_dft_sub_negate_inplace => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_negate_inplace,
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_consume,
            test_vec_znx_idft_apply_tmpa => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_tmpa,
        }
    }
    cross_backend_test_suite! {
        mod vmp,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT120Avx,
        size = 1<<8,
        base2k = 50,
        tests = {
            test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
            test_vmp_apply_dft_to_dft_add => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_add,
        }
    }

    backend_test_suite! {
        mod sampling,
        backend = crate::NTT120Avx,
        size = 1<<12,
        tests = {
            test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
            test_vec_znx_fill_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_normal,
            test_vec_znx_add_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_normal,
        }
    }

    // NTT CHANGE_MODE_N boundary tests.
    // CHANGE_MODE_N = 1024: for n <= 1024 the AVX NTT runs fully by-block;
    // for n > 1024 it first completes upper levels by-level then switches to
    // by-block for the remaining levels. These suites ensure both modes are
    // exercised and agree with the reference backend.

    // n = 1024: last size that uses by-block only.
    cross_backend_test_suite! {
        mod ntt_n1024,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT120Avx,
        size = 1<<10,
        base2k = 50,
        tests = {
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_consume,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        }
    }

    // n = 2048: first size that uses the by-level phase.
    cross_backend_test_suite! {
        mod ntt_n2048,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT120Avx,
        size = 1<<11,
        base2k = 50,
        tests = {
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_consume,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        }
    }

    // n = 65536: large size exercising many by-level stages.
    cross_backend_test_suite! {
        mod ntt_n65536,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT120Avx,
        size = 1<<16,
        base2k = 50,
        tests = {
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_consume,
        }
    }
}

#[test]
fn test_convolution_by_const_fft64_avx() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(8);
    test_convolution_by_const(&module);
}

#[test]
fn test_convolution_fft64_avx() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(8);
    test_convolution(&module);
}

#[test]
fn test_convolution_pairwise_fft64_avx() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(8);
    test_convolution_pairwise(&module);
}
