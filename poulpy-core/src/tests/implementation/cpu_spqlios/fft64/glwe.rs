use poulpy_backend::cpu_spqlios::FFT64;
use poulpy_hal::{api::ModuleNew, layouts::Module};

use crate::tests::generics::{
    automorphism::{test_glwe_automorphism, test_glwe_automorphism_inplace},
    encryption::{test_glwe_compressed_encrypt_sk, test_glwe_encrypt_pk, test_glwe_encrypt_sk, test_glwe_encrypt_zero_sk},
    external_product::{test_glwe_external_product, test_glwe_external_product_inplace},
    keyswitch::{test_glwe_keyswitch, test_glwe_keyswitch_inplace},
    test_glwe_packing, test_glwe_trace_inplace,
};

#[test]
fn glwe_encrypt_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test_glwe_encrypt_sk rank: {}", rank);
        test_glwe_encrypt_sk(&module, 8, 54, 30, rank);
    });
}

#[test]
fn glwe_compressed_encrypt_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test_glwe_compressed_encrypt_sk rank: {}", rank);
        test_glwe_compressed_encrypt_sk(&module, 8, 54, 30, rank);
    });
}

#[test]
fn glwe_encrypt_zero_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test_glwe_encrypt_zero_sk rank: {}", rank);
        test_glwe_encrypt_zero_sk(&module, 8, 64, rank);
    });
}

#[test]
fn glwe_encrypt_pk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test_glwe_encrypt_pk rank: {}", rank);
        test_glwe_encrypt_pk(&module, 8, 64, 64, rank)
    });
}

#[test]
fn glwe_keyswitch() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_in: usize = 45;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank_in| {
        (1..4).for_each(|rank_out| {
            (1..digits + 1).for_each(|di| {
                let k_ksk: usize = k_in + basek * di;
                let k_out: usize = k_ksk; // better capture noise
                println!(
                    "test_glwe_keyswitch digits: {} rank_in: {} rank_out: {}",
                    di, rank_in, rank_out
                );
                test_glwe_keyswitch(&module, basek, k_out, k_in, k_ksk, di, rank_in, rank_out);
            })
        });
    });
}

#[test]
fn glwe_keyswitch_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 45;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_ct + basek * di;
            println!("test_glwe_keyswitch_inplace digits: {} rank: {}", di, rank);
            test_glwe_keyswitch_inplace(&module, basek, k_ct, k_ksk, di, rank);
        });
    });
}

#[test]
fn glwe_automorphism() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_in: usize = 60;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_in + basek * di;
            let k_out: usize = k_ksk; // Better capture noise.
            println!("test_glwe_automorphism digits: {} rank: {}", di, rank);
            test_glwe_automorphism(&module, basek, -5, k_out, k_in, k_ksk, di, rank);
        })
    });
}

#[test]
fn glwe_automorphism_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_ct + basek * di;
            println!(
                "test_glwe_automorphism_inplace digits: {} rank: {}",
                di, rank
            );
            test_glwe_automorphism_inplace(&module, basek, -5, k_ct, k_ksk, di, rank);
        });
    });
}

#[test]
fn glwe_external_product() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_in: usize = 45;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ggsw: usize = k_in + basek * di;
            let k_out: usize = k_ggsw; // Better capture noise
            println!("test_glwe_external_product digits: {} rank: {}", di, rank);
            test_glwe_external_product(&module, basek, k_out, k_in, k_ggsw, di, rank);
        });
    });
}

#[test]
fn glwe_external_product_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ggsw: usize = k_ct + basek * di;
            println!(
                "test_glwe_external_product_inplace digits: {} rank: {}",
                di, rank
            );
            test_glwe_external_product_inplace(&module, basek, k_ct, k_ggsw, di, rank);
        });
    });
}

#[test]
fn glwe_trace_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test_glwe_trace_inplace rank: {}", rank);
        test_glwe_trace_inplace(&module, 8, 54, rank);
    });
}

#[test]
fn glwe_packing() {
    let log_n: usize = 5;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    test_glwe_packing(&module);
}
