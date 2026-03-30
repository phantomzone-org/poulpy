use poulpy_cpu_avx::NTT120Avx;
use poulpy_hal::{api::ModuleNew, layouts::Module};

use crate::leveled::tests::test_suite::{
    NTT120_PARAMS,
    arithmetic::{
        test_add_const_ct, test_add_ct_ct, test_add_pt_ct, test_neg_ct, test_sub_const_ct, test_sub_ct_ct, test_sub_pt_ct,
    },
    drop_level::{test_drop_bits_crosslimb, test_drop_bits_sublimb, test_drop_limbs},
    encrypt_decrypt::test_encrypt_decrypt,
    precision::test_precision,
};

#[test]
fn encrypt_decrypt() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(65536);
    test_encrypt_decrypt(&module, NTT120_PARAMS);
}

#[test]
fn drop_limbs() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(65536);
    test_drop_limbs(&module, NTT120_PARAMS);
}

#[test]
fn drop_bits_sublimb() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(65536);
    test_drop_bits_sublimb(&module, NTT120_PARAMS);
}

#[test]
fn drop_bits_crosslimb() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(65536);
    test_drop_bits_crosslimb(&module, NTT120_PARAMS);
}

#[test]
fn add_ct_ct() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(65536);
    test_add_ct_ct(&module, NTT120_PARAMS);
}

#[test]
fn add_pt_ct() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(65536);
    test_add_pt_ct(&module, NTT120_PARAMS);
}

#[test]
fn add_const_ct() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(65536);
    test_add_const_ct(&module, NTT120_PARAMS);
}

#[test]
fn sub_ct_ct() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(65536);
    test_sub_ct_ct(&module, NTT120_PARAMS);
}

#[test]
fn sub_pt_ct() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(65536);
    test_sub_pt_ct(&module, NTT120_PARAMS);
}

#[test]
fn sub_const_ct() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(65536);
    test_sub_const_ct(&module, NTT120_PARAMS);
}

#[test]
fn neg_ct() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(65536);
    test_neg_ct(&module, NTT120_PARAMS);
}

#[test]
fn precision() {
    let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(65536);
    test_precision(&module, NTT120_PARAMS);
}
