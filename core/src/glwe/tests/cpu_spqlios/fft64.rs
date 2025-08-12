use backend::{
    hal::{api::ModuleNew, layouts::Module},
    implementation::cpu_spqlios::FFT64,
};

use crate::glwe::tests::{
    generic_automorphism::{test_automorphism, test_automorphism_inplace},
    generic_encryption::{test_encrypt_pk, test_encrypt_sk, test_encrypt_sk_compressed, test_encrypt_zero_sk},
    generic_external_product::{test_external_product, test_external_product_inplace},
    generic_keyswitch::{test_keyswitch, test_keyswitch_inplace},
    generic_serialization::{test_serialization, test_serialization_compressed},
    packing::test_packing,
    trace::test_trace_inplace,
};

#[test]
fn encrypt_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test encrypt_sk rank: {}", rank);
        test_encrypt_sk(&module, 8, 54, 30, 3.2, rank);
    });
}

#[test]
fn encrypt_sk_compressed() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test encrypt_sk rank: {}", rank);
        test_encrypt_sk_compressed(&module, 8, 54, 30, 3.2, rank);
    });
}

#[test]
fn encrypt_zero_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test encrypt_zero_sk rank: {}", rank);
        test_encrypt_zero_sk(&module, 8, 64, 3.2, rank);
    });
}

#[test]
fn encrypt_pk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test encrypt_pk rank: {}", rank);
        test_encrypt_pk(&module, 8, 64, 64, 3.2, rank)
    });
}

#[test]
fn apply() {
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
                    "test keyswitch digits: {} rank_in: {} rank_out: {}",
                    di, rank_in, rank_out
                );
                test_keyswitch(
                    &module, basek, k_out, k_in, k_ksk, di, rank_in, rank_out, 3.2,
                );
            })
        });
    });
}

#[test]
fn apply_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 45;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_ct + basek * di;
            println!("test keyswitch_inplace digits: {} rank: {}", di, rank);
            test_keyswitch_inplace(&module, basek, k_ct, k_ksk, di, rank, 3.2);
        });
    });
}

#[test]
fn automorphism_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_ct + basek * di;
            println!("test automorphism_inplace digits: {} rank: {}", di, rank);
            test_automorphism_inplace(&module, basek, -5, k_ct, k_ksk, di, rank, 3.2);
        });
    });
}

#[test]
fn automorphism() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_in: usize = 60;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_in + basek * di;
            let k_out: usize = k_ksk; // Better capture noise.
            println!("test automorphism digits: {} rank: {}", di, rank);
            test_automorphism(&module, basek, -5, k_out, k_in, k_ksk, di, rank, 3.2);
        })
    });
}

#[test]
fn external_product() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_in: usize = 45;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ggsw: usize = k_in + basek * di;
            let k_out: usize = k_ggsw; // Better capture noise
            println!("test external_product digits: {} rank: {}", di, rank);
            test_external_product(&module, basek, k_out, k_in, k_ggsw, di, rank, 3.2);
        });
    });
}

#[test]
fn external_product_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ggsw: usize = k_ct + basek * di;
            println!("test external_product digits: {} rank: {}", di, rank);
            test_external_product_inplace(&module, basek, k_ct, k_ggsw, di, rank, 3.2);
        });
    });
}

#[test]
fn trace_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test trace_inplace rank: {}", rank);
        test_trace_inplace(&module, 8, 54, 3.2, rank);
    });
}

#[test]
fn packing() {
    let log_n: usize = 5;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    test_packing(&module);
}

#[test]
fn serialization() {
    let log_n: usize = 5;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    test_serialization(&module);
}

#[test]
fn serialization_compressed() {
    let log_n: usize = 5;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    test_serialization_compressed(&module);
}
