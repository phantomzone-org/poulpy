use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::{api::ModuleNew, layouts::Module};

use crate::leveled::tests::test_suite::{
    arithmetic::{test_add_cleartext_ct, test_add_ct_ct, test_add_pt_ct},
    drop_level::{test_drop_bits_crosslimb, test_drop_bits_sublimb, test_drop_limbs},
    encrypt_decrypt::test_encrypt_decrypt,
};

#[test]
fn encrypt_decrypt() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_encrypt_decrypt(&module);
}

#[test]
fn drop_limbs() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_drop_limbs(&module);
}

#[test]
fn drop_bits_sublimb() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_drop_bits_sublimb(&module);
}

#[test]
fn drop_bits_crosslimb() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_drop_bits_crosslimb(&module);
}

#[test]
fn add_ct_ct() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_add_ct_ct(&module);
}

#[test]
fn add_pt_ct() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_add_pt_ct(&module);
}

#[test]
fn add_cleartext_ct() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_add_cleartext_ct(&module);
}
