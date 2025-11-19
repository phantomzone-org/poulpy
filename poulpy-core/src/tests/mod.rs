pub mod test_suite;

#[cfg(test)]
mod serialization;

#[cfg(test)]
mod poulpy_core {
    use poulpy_hal::backend_test_suite;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    backend_test_suite!(
    mod cpu_avx,
    backend = poulpy_cpu_avx::FFT64Avx,
    size = 1<<8,
    tests = {
        //GLWE Encryption
        glwe_encrypt_sk => crate::tests::test_suite::encryption::test_glwe_encrypt_sk,
        glwe_compressed_encrypt_sk => crate::tests::test_suite::encryption::test_glwe_compressed_encrypt_sk,
        glwe_encrypt_zero_sk => crate::tests::test_suite::encryption::test_glwe_encrypt_zero_sk,
        glwe_encrypt_pk => crate::tests::test_suite::encryption::test_glwe_encrypt_pk,
        // GLWE Base2k Conversion
        glwe_base2k_conv => crate::tests::test_suite::test_glwe_base2k_conversion,
        // GLWE Keyswitch
        glwe_keyswitch => crate::tests::test_suite::keyswitch::test_glwe_keyswitch,
        glwe_keyswitch_inplace => crate::tests::test_suite::keyswitch::test_glwe_keyswitch_inplace,
        // GLWE Automorphism
        glwe_automorphism => crate::tests::test_suite::automorphism::test_glwe_automorphism,
        glwe_automorphism_inplace => crate::tests::test_suite::automorphism::test_glwe_automorphism_inplace,
        // GLWE External Product
        glwe_external_product => crate::tests::test_suite::external_product::test_glwe_external_product,
        glwe_external_product_inplace => crate::tests::test_suite::external_product::test_glwe_external_product_inplace,
        // GLWE Trace
        glwe_trace_inplace => crate::tests::test_suite::test_glwe_trace_inplace,
        glwe_packing => crate::tests::test_suite::test_glwe_packing,
        glwe_packer => crate::tests::test_suite::test_glwe_packer,
        // GGLWE Encryption
        gglwe_switching_key_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_switching_key_encrypt_sk,
        gglwe_switching_key_compressed_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_switching_key_compressed_encrypt_sk,
        gglwe_compressed_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_compressed_encrypt_sk,
        gglwe_automorphism_key_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_automorphism_key_encrypt_sk,
        gglwe_automorphism_key_compressed_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_automorphism_key_compressed_encrypt_sk,
        gglwe_tensor_key_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_tensor_key_encrypt_sk,
        gglwe_tensor_key_compressed_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_tensor_key_compressed_encrypt_sk,
        gglwe_to_ggsw_key_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_to_ggsw_key_encrypt_sk,
        // GGLWE Keyswitching
        gglwe_switching_key_keyswitch => crate::tests::test_suite::keyswitch::test_gglwe_switching_key_keyswitch,
        gglwe_switching_key_keyswitch_inplace => crate::tests::test_suite::keyswitch::test_gglwe_switching_key_keyswitch_inplace,
        // GGLWE External Product
        gglwe_switching_key_external_product => crate::tests::test_suite::external_product::test_gglwe_switching_key_external_product,
        gglwe_switching_key_external_product_inplace => crate::tests::test_suite::external_product::test_gglwe_switching_key_external_product_inplace,
        // GGLWE Automorphism
        gglwe_automorphism_key_automorphism => crate::tests::test_suite::automorphism::test_gglwe_automorphism_key_automorphism,
        gglwe_automorphism_key_automorphism_inplace => crate::tests::test_suite::automorphism::test_gglwe_automorphism_key_automorphism_inplace,
        // GGSW Encryption
        ggsw_encrypt_sk => crate::tests::test_suite::encryption::test_ggsw_encrypt_sk,
        ggsw_compressed_encrypt_sk => crate::tests::test_suite::encryption::test_ggsw_compressed_encrypt_sk,
        // GGSW Keyswitching
        ggsw_keyswitch => crate::tests::test_suite::keyswitch::test_ggsw_keyswitch,
        ggsw_keyswitch_inplace => crate::tests::test_suite::keyswitch::test_ggsw_keyswitch_inplace,
        // GGSW External Product
        ggsw_external_product => crate::tests::test_suite::external_product::test_ggsw_external_product,
        ggsw_external_product_inplace => crate::tests::test_suite::external_product::test_ggsw_external_product_inplace,
        // GGSW Automorphism
        ggsw_automorphism => crate::tests::test_suite::automorphism::test_ggsw_automorphism,
        ggsw_automorphism_inplace => crate::tests::test_suite::automorphism::test_ggsw_automorphism_inplace,
        // LWE
        lwe_keyswitch => crate::tests::test_suite::keyswitch::test_lwe_keyswitch,
        glwe_to_lwe => crate::tests::test_suite::test_glwe_to_lwe,
        lwe_to_glwe => crate::tests::test_suite::test_lwe_to_glwe,
    }
    );

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    backend_test_suite!(
    mod cpu_ref,
    backend = poulpy_cpu_ref::FFT64Ref,
    size = 1<<8,
    tests = {
        //GLWE Encryption
        glwe_encrypt_sk => crate::tests::test_suite::encryption::test_glwe_encrypt_sk,
        glwe_compressed_encrypt_sk => crate::tests::test_suite::encryption::test_glwe_compressed_encrypt_sk,
        glwe_encrypt_zero_sk => crate::tests::test_suite::encryption::test_glwe_encrypt_zero_sk,
        glwe_encrypt_pk => crate::tests::test_suite::encryption::test_glwe_encrypt_pk,
        // GLWE Base2k Conversion
        glwe_base2k_conv => crate::tests::test_suite::test_glwe_base2k_conversion,
        // GLWE Keyswitch
        glwe_keyswitch => crate::tests::test_suite::keyswitch::test_glwe_keyswitch,
        glwe_keyswitch_inplace => crate::tests::test_suite::keyswitch::test_glwe_keyswitch_inplace,
        // GLWE Automorphism
        glwe_automorphism => crate::tests::test_suite::automorphism::test_glwe_automorphism,
        glwe_automorphism_inplace => crate::tests::test_suite::automorphism::test_glwe_automorphism_inplace,
        // GLWE External Product
        glwe_external_product => crate::tests::test_suite::external_product::test_glwe_external_product,
        glwe_external_product_inplace => crate::tests::test_suite::external_product::test_glwe_external_product_inplace,
        // GLWE Trace
        glwe_trace_inplace => crate::tests::test_suite::test_glwe_trace_inplace,
        glwe_packing => crate::tests::test_suite::test_glwe_packing,
        glwe_packer => crate::tests::test_suite::test_glwe_packer,
        // GGLWE Encryption
        gglwe_switching_key_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_switching_key_encrypt_sk,
        gglwe_switching_key_compressed_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_switching_key_compressed_encrypt_sk,
        gglwe_compressed_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_compressed_encrypt_sk,
        gglwe_automorphism_key_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_automorphism_key_encrypt_sk,
        gglwe_automorphism_key_compressed_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_automorphism_key_compressed_encrypt_sk,
        gglwe_tensor_key_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_tensor_key_encrypt_sk,
        gglwe_tensor_key_compressed_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_tensor_key_compressed_encrypt_sk,
        gglwe_to_ggsw_key_encrypt_sk => crate::tests::test_suite::encryption::test_gglwe_to_ggsw_key_encrypt_sk,
        // GGLWE Keyswitching
        gglwe_switching_key_keyswitch => crate::tests::test_suite::keyswitch::test_gglwe_switching_key_keyswitch,
        gglwe_switching_key_keyswitch_inplace => crate::tests::test_suite::keyswitch::test_gglwe_switching_key_keyswitch_inplace,
        // GGLWE External Product
        gglwe_switching_key_external_product => crate::tests::test_suite::external_product::test_gglwe_switching_key_external_product,
        gglwe_switching_key_external_product_inplace => crate::tests::test_suite::external_product::test_gglwe_switching_key_external_product_inplace,
        // GGLWE Automorphism
        gglwe_automorphism_key_automorphism => crate::tests::test_suite::automorphism::test_gglwe_automorphism_key_automorphism,
        gglwe_automorphism_key_automorphism_inplace => crate::tests::test_suite::automorphism::test_gglwe_automorphism_key_automorphism_inplace,
        // GGSW Encryption
        ggsw_encrypt_sk => crate::tests::test_suite::encryption::test_ggsw_encrypt_sk,
        ggsw_compressed_encrypt_sk => crate::tests::test_suite::encryption::test_ggsw_compressed_encrypt_sk,
        // GGSW Keyswitching
        ggsw_keyswitch => crate::tests::test_suite::keyswitch::test_ggsw_keyswitch,
        ggsw_keyswitch_inplace => crate::tests::test_suite::keyswitch::test_ggsw_keyswitch_inplace,
        // GGSW External Product
        ggsw_external_product => crate::tests::test_suite::external_product::test_ggsw_external_product,
        ggsw_external_product_inplace => crate::tests::test_suite::external_product::test_ggsw_external_product_inplace,
        // GGSW Automorphism
        ggsw_automorphism => crate::tests::test_suite::automorphism::test_ggsw_automorphism,
        ggsw_automorphism_inplace => crate::tests::test_suite::automorphism::test_ggsw_automorphism_inplace,
        // LWE
        lwe_keyswitch => crate::tests::test_suite::keyswitch::test_lwe_keyswitch,
        glwe_to_lwe => crate::tests::test_suite::test_glwe_to_lwe,
        lwe_to_glwe => crate::tests::test_suite::test_lwe_to_glwe,
    }
    );
}
