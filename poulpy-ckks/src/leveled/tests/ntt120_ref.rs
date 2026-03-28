use poulpy_cpu_ref::{FFT64Ref, NTT120Ref};
use poulpy_hal::{api::ModuleNew, layouts::Module};

use crate::leveled::tests::test_suite::{
    arithmetic::{
        test_add_const_ct, test_add_ct_ct, test_add_pt_ct, test_neg_ct, test_sub_const_ct, test_sub_ct_ct, test_sub_pt_ct,
    },
    drop_level::{test_drop_bits_crosslimb, test_drop_bits_sublimb, test_drop_limbs},
    encrypt_decrypt::test_encrypt_decrypt,
};

#[test]
fn encrypt_decrypt() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(32768);
    let codec: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_encrypt_decrypt(&module, &codec);
}

#[test]
fn drop_limbs() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(32768);
    let codec: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_drop_limbs(&module, &codec);
}

#[test]
fn drop_bits_sublimb() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(32768);
    let codec: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_drop_bits_sublimb(&module, &codec);
}

#[test]
fn drop_bits_crosslimb() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(32768);
    let codec: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_drop_bits_crosslimb(&module, &codec);
}

#[test]
fn add_ct_ct() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(32768);
    let codec: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_add_ct_ct(&module, &codec);
}

#[test]
fn add_pt_ct() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(32768);
    let codec: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_add_pt_ct(&module, &codec);
}

#[test]
fn add_const_ct() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(32768);
    let codec: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_add_const_ct(&module, &codec);
}

#[test]
fn sub_ct_ct() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(32768);
    let codec: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_sub_ct_ct(&module, &codec);
}

#[test]
fn sub_pt_ct() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(32768);
    let codec: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_sub_pt_ct(&module, &codec);
}

#[test]
fn sub_const_ct() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(32768);
    let codec: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_sub_const_ct(&module, &codec);
}

#[test]
fn neg_ct() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(32768);
    let codec: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_neg_ct(&module, &codec);
}
