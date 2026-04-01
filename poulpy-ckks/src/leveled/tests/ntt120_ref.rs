use poulpy_cpu_ref::NTT120Ref;
use poulpy_hal::{api::ModuleNew, layouts::Module};

use crate::leveled::tests::test_suite::{
    NTT120_PARAMS,
    add::{test_add, test_add_const, test_add_pt},
    encryption::test_encrypt_decrypt,
    level::{test_drop_bits_crosslimb, test_drop_bits_sublimb, test_drop_limbs},
    mul::test_mul,
    neg::test_neg,
    precision::test_precision,
    sub::{test_sub, test_sub_const, test_sub_pt},
};

#[test]
fn encrypt_decrypt() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(NTT120_PARAMS.n as u64);
    test_encrypt_decrypt(&module, NTT120_PARAMS);
}

#[test]
fn drop_limbs() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(NTT120_PARAMS.n as u64);
    test_drop_limbs(&module, NTT120_PARAMS);
}

#[test]
fn drop_bits_sublimb() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(NTT120_PARAMS.n as u64);
    test_drop_bits_sublimb(&module, NTT120_PARAMS);
}

#[test]
fn drop_bits_crosslimb() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(NTT120_PARAMS.n as u64);
    test_drop_bits_crosslimb(&module, NTT120_PARAMS);
}

#[test]
fn add() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(NTT120_PARAMS.n as u64);
    test_add(&module, NTT120_PARAMS);
}

#[test]
fn add_pt() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(NTT120_PARAMS.n as u64);
    test_add_pt(&module, NTT120_PARAMS);
}

#[test]
fn add_const() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(NTT120_PARAMS.n as u64);
    test_add_const(&module, NTT120_PARAMS);
}

#[test]
fn sub() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(NTT120_PARAMS.n as u64);
    test_sub(&module, NTT120_PARAMS);
}

#[test]
fn sub_pt() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(NTT120_PARAMS.n as u64);
    test_sub_pt(&module, NTT120_PARAMS);
}

#[test]
fn sub_const() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(NTT120_PARAMS.n as u64);
    test_sub_const(&module, NTT120_PARAMS);
}

#[test]
fn neg() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(NTT120_PARAMS.n as u64);
    test_neg(&module, NTT120_PARAMS);
}

#[test]
fn mul() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(NTT120_PARAMS.n as u64);
    test_mul(&module, NTT120_PARAMS);
}

#[test]
fn precision() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(NTT120_PARAMS.n as u64);
    test_precision(&module, NTT120_PARAMS);
}
