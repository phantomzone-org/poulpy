use backend::{
    hal::{api::ModuleNew, layouts::Module},
    implementation::cpu_spqlios::FFT64,
};

use crate::tfhe::blind_rotation::tests::{
    generic_blind_rotation::test_blind_rotation,
    generic_lut::{test_lut_extended, test_lut_standard},
};

#[test]
fn lut_standard() {
    let module: Module<FFT64> = Module::<FFT64>::new(32);
    test_lut_standard(&module);
}

#[test]
fn lut_extended() {
    let module: Module<FFT64> = Module::<FFT64>::new(32);
    test_lut_extended(&module);
}

#[test]
fn standard() {
    let module: Module<FFT64> = Module::<FFT64>::new(512);
    test_blind_rotation(&module, 224, 1, 1);
}

#[test]
fn block_binary() {
    let module: Module<FFT64> = Module::<FFT64>::new(512);
    test_blind_rotation(&module, 224, 7, 1);
}

#[test]
fn block_binary_extended() {
    let module: Module<FFT64> = Module::<FFT64>::new(512);
    test_blind_rotation(&module, 224, 7, 2);
}
