use backend::{FFT64, Module, ModuleNew};

use crate::gglwe::test_fft64::gglwe_generic::{
    test_encrypt_sk, test_external_product, test_external_product_inplace, test_keyswitch, test_keyswitch_inplace,
};

#[test]
fn encrypt_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ksk: usize = 54;
    let digits: usize = k_ksk / basek;
    (1..4).for_each(|rank_in| {
        (1..4).for_each(|rank_out| {
            (1..digits + 1).for_each(|di| {
                println!(
                    "test encrypt_sk digits: {} ranks: ({} {})",
                    di, rank_in, rank_out
                );
                test_encrypt_sk(&module, basek, k_ksk, di, rank_in, rank_out, 3.2);
            });
        });
    });
}

#[test]
fn keyswitch() {
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
                        "test key_switch digits: {} ranks: ({},{},{})",
                        di, rank_in_s0s1, rank_out_s0s1, rank_out_s1s2
                    );
                    let k_out: usize = k_ksk; // Better capture noise.
                    test_keyswitch(
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
fn keyswitch_inplace() {
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
                    "test key_switch_inplace digits: {} ranks: ({},{})",
                    di, rank_in_s0s1, rank_out_s0s1
                );
                test_keyswitch_inplace(
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
fn external_product() {
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
                    "test external_product digits: {} ranks: ({} {})",
                    di, rank_in, rank_out
                );
                let k_out: usize = k_in; // Better capture noise.
                test_external_product(
                    &module, basek, k_out, k_in, k_ggsw, di, rank_in, rank_out, 3.2,
                );
            });
        });
    });
}

#[test]
fn external_product_inplace() {
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
                    "test external_product_inplace digits: {} ranks: ({} {})",
                    di, rank_in, rank_out
                );
                test_external_product_inplace(&module, basek, k_ct, k_ggsw, di, rank_in, rank_out, 3.2);
            });
        });
    });
}
