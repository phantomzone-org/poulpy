use poulpy_backend::{
    hal::{api::ModuleNew, layouts::Module},
    implementation::cpu_spqlios::FFT64,
};

use crate::tests::generics::{
    automorphism::{test_gglwe_automorphism_key_automorphism, test_gglwe_automorphism_key_automorphism_inplace},
    encryption::{
        test_gglwe_automorphisk_key_compressed_encrypt_sk, test_gglwe_automorphisk_key_encrypt_sk,
        test_gglwe_switching_key_compressed_encrypt_sk, test_gglwe_switching_key_encrypt_sk,
        test_glwe_tensor_key_compressed_encrypt_sk, test_glwe_tensor_key_encrypt_sk,
    },
    external_product::{test_gglwe_switching_key_external_product, test_gglwe_switching_key_external_product_inplace},
    keyswitch::{test_gglwe_switching_key_keyswitch, test_gglwe_switching_key_keyswitch_inplace},
};

#[test]
fn gglwe_switching_key_encrypt_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ksk: usize = 54;
    let digits: usize = k_ksk / basek;
    (1..4).for_each(|rank_in| {
        (1..4).for_each(|rank_out| {
            (1..digits + 1).for_each(|di| {
                println!(
                    "test_gglwe_switching_key_encrypt_sk digits: {} ranks: ({} {})",
                    di, rank_in, rank_out
                );
                test_gglwe_switching_key_encrypt_sk(&module, basek, k_ksk, di, rank_in, rank_out, 3.2);
            });
        });
    });
}

#[test]
fn gglwe_switching_key_compressed_encrypt_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ksk: usize = 54;
    let digits: usize = k_ksk / basek;
    (1..4).for_each(|rank_in| {
        (1..4).for_each(|rank_out| {
            (1..digits + 1).for_each(|di| {
                println!(
                    "test_gglwe_switching_key_compressed_encrypt_sk digits: {} ranks: ({} {})",
                    di, rank_in, rank_out
                );
                test_gglwe_switching_key_compressed_encrypt_sk(&module, basek, k_ksk, di, rank_in, rank_out, 3.2);
            });
        });
    });
}

#[test]
fn gglwe_switching_key_keyswitch() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_in: usize = 60;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank_in_s0s1| {
        (1..4).for_each(|rank_out_s0s1| {
            (1..4).for_each(|rank_out_s1s2| {
                (1..digits + 1).for_each(|di| {
                    let k_ksk: usize = k_in + basek * di;
                    println!(
                        "test_gglwe_switching_key_keyswitch digits: {} ranks: ({},{},{})",
                        di, rank_in_s0s1, rank_out_s0s1, rank_out_s1s2
                    );
                    let k_out: usize = k_ksk; // Better capture noise.
                    test_gglwe_switching_key_keyswitch(
                        &module,
                        basek,
                        k_out,
                        k_in,
                        k_ksk,
                        di,
                        rank_in_s0s1,
                        rank_out_s0s1,
                        rank_out_s1s2,
                        3.2,
                    );
                })
            })
        });
    });
}

#[test]
fn gglwe_switching_key_keyswitch_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank_in_s0s1| {
        (1..4).for_each(|rank_out_s0s1| {
            (1..digits + 1).for_each(|di| {
                let k_ksk: usize = k_ct + basek * di;
                println!(
                    "test_gglwe_switching_key_keyswitch_inplace digits: {} ranks: ({},{})",
                    di, rank_in_s0s1, rank_out_s0s1
                );
                test_gglwe_switching_key_keyswitch_inplace(
                    &module,
                    basek,
                    k_ct,
                    k_ksk,
                    di,
                    rank_in_s0s1,
                    rank_out_s0s1,
                    3.2,
                );
            });
        });
    });
}

#[test]
fn gglwe_switching_key_external_product() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_in: usize = 60;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank_in| {
        (1..4).for_each(|rank_out| {
            (1..digits + 1).for_each(|di| {
                let k_ggsw: usize = k_in + basek * di;
                println!(
                    "test_gglwe_switching_key_external_product digits: {} ranks: ({} {})",
                    di, rank_in, rank_out
                );
                let k_out: usize = k_in; // Better capture noise.
                test_gglwe_switching_key_external_product(
                    &module, basek, k_out, k_in, k_ggsw, di, rank_in, rank_out, 3.2,
                );
            });
        });
    });
}

#[test]
fn gglwe_switching_key_external_product_inplace() {
    let log_n: usize = 5;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank_in| {
        (1..4).for_each(|rank_out| {
            (1..digits).for_each(|di| {
                let k_ggsw: usize = k_ct + basek * di;
                println!(
                    "test_gglwe_switching_key_external_product_inplace digits: {} ranks: ({} {})",
                    di, rank_in, rank_out
                );
                test_gglwe_switching_key_external_product_inplace(&module, basek, k_ct, k_ggsw, di, rank_in, rank_out, 3.2);
            });
        });
    });
}

#[test]
fn gglwe_automorphisk_key_encrypt_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k: usize = 60;
    let digits: usize = k.div_ceil(basek) - 1;
    let sigma: f64 = 3.2;
    (1..4).for_each(|rank| {
        (2..digits + 1).for_each(|di| {
            println!(
                "test_gglwe_automorphisk_key_encrypt_sk digits: {} rank: {}",
                di, rank
            );
            test_gglwe_automorphisk_key_encrypt_sk(&module, basek, k, di, rank, sigma);
        });
    });
}

#[test]
fn gglwe_automorphisk_key_compressed_encrypt_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k: usize = 60;
    let digits: usize = k.div_ceil(basek) - 1;
    let sigma: f64 = 3.2;
    (1..4).for_each(|rank| {
        (2..digits + 1).for_each(|di| {
            println!(
                "test_gglwe_automorphisk_key_compressed_encrypt_sk digits: {} rank: {}",
                di, rank
            );
            test_gglwe_automorphisk_key_compressed_encrypt_sk(&module, basek, k, di, rank, sigma);
        });
    });
}

#[test]
fn gglwe_automorphism_key_automorphism() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_in: usize = 60;
    let k_out: usize = 40;
    let digits: usize = k_in.div_ceil(basek);
    let sigma: f64 = 3.2;
    (1..4).for_each(|rank| {
        (2..digits + 1).for_each(|di| {
            println!(
                "test_gglwe_automorphism_key_automorphism: {} rank: {}",
                di, rank
            );
            let k_apply: usize = (digits + di) * basek;
            test_gglwe_automorphism_key_automorphism(&module, -1, 5, basek, di, k_in, k_out, k_apply, sigma, rank);
        });
    });
}

#[test]
fn gglwe_automorphism_key_automorphism_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_in: usize = 60;
    let digits: usize = k_in.div_ceil(basek);
    let sigma: f64 = 3.2;
    (1..4).for_each(|rank| {
        (2..digits + 1).for_each(|di| {
            println!(
                "test_gglwe_automorphism_key_automorphism_inplace: {} rank: {}",
                di, rank
            );
            let k_apply: usize = (digits + di) * basek;
            test_gglwe_automorphism_key_automorphism_inplace(&module, -1, 5, basek, di, k_in, k_apply, sigma, rank);
        });
    });
}

#[test]
fn glwe_tensor_key_encrypt_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test_glwe_tensor_key_encrypt_sk rank: {}", rank);
        test_glwe_tensor_key_encrypt_sk(&module, 16, 54, 3.2, rank);
    });
}

#[test]
fn glwe_tensor_key_compressed_encrypt_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test_glwe_tensor_key_compressed_encrypt_sk rank: {}", rank);
        test_glwe_tensor_key_compressed_encrypt_sk(&module, 16, 54, 3.2, rank);
    });
}
