use poulpy_backend::{
    hal::{api::ModuleNew, layouts::Module},
    implementation::cpu_spqlios::FFT64,
};

use crate::tests::generics::{
    automorphism::{test_ggsw_automorphism, test_ggsw_automorphism_inplace},
    encryption::{test_ggsw_compressed_encrypt_sk, test_ggsw_encrypt_sk},
    external_product::{test_ggsw_external_product, test_ggsw_external_product_inplace},
    keyswitch::{test_ggsw_keyswitch, test_ggsw_keyswitch_inplace},
};

#[test]
fn ggsw_encrypt_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 54;
    let digits: usize = k_ct / basek;
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            println!("test_ggsw_encrypt_sk digits: {} rank: {}", di, rank);
            test_ggsw_encrypt_sk(&module, basek, k_ct, di, rank, 3.2);
        });
    });
}

#[test]
fn ggsw_compressed_encrypt_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 54;
    let digits: usize = k_ct / basek;
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            println!(
                "test_ggsw_compressed_encrypt_sk digits: {} rank: {}",
                di, rank
            );
            test_ggsw_compressed_encrypt_sk(&module, basek, k_ct, di, rank, 3.2);
        });
    });
}

#[test]
fn ggsw_keyswitch() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_in: usize = 54;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_in + basek * di;
            let k_tsk: usize = k_ksk;
            println!("test_ggsw_keyswitch digits: {} rank: {}", di, rank);
            let k_out: usize = k_ksk; // Better capture noise.
            test_ggsw_keyswitch(&module, basek, k_out, k_in, k_ksk, k_tsk, di, rank, 3.2);
        });
    });
}

#[test]
fn ggsw_keyswitch_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 54;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_ct + basek * di;
            let k_tsk: usize = k_ksk;
            println!("test_ggsw_keyswitch_inplace digits: {} rank: {}", di, rank);
            test_ggsw_keyswitch_inplace(&module, basek, k_ct, k_ksk, k_tsk, di, rank, 3.2);
        });
    });
}

#[test]
fn ggsw_automorphism() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_in: usize = 54;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_in + basek * di;
            let k_tsk: usize = k_ksk;
            println!("test_ggsw_automorphism rank: {}", rank);
            let k_out: usize = k_ksk; // Better capture noise.
            test_ggsw_automorphism(-5, &module, basek, k_out, k_in, k_ksk, k_tsk, di, rank, 3.2);
        });
    });
}

#[test]
fn ggsw_automorphism_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 54;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_ct + basek * di;
            let k_tsk: usize = k_ksk;
            println!("test_ggsw_automorphism_inplace rank: {}", rank);
            test_ggsw_automorphism_inplace(-5, &module, basek, k_ct, k_ksk, k_tsk, di, rank, 3.2);
        });
    });
}

#[test]
fn ggsw_external_product() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_in: usize = 60;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ggsw: usize = k_in + basek * di;
            println!("test external_product digits: {} ranks: {}", di, rank);
            let k_out: usize = k_in; // Better capture noise.
            test_ggsw_external_product(&module, basek, k_in, k_out, k_ggsw, di, rank, 3.2);
        });
    });
}

#[test]
fn ggsw_external_product_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits).for_each(|di| {
            let k_ggsw: usize = k_ct + basek * di;
            println!(
                "test_ggsw_external_product_inplace digits: {} rank: {}",
                di, rank
            );
            test_ggsw_external_product_inplace(&module, basek, k_ct, k_ggsw, di, rank, 3.2);
        });
    });
}
