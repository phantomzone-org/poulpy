use poulpy_backend::{
    hal::{api::ModuleNew, layouts::Module},
    implementation::cpu_spqlios::FFT64,
};

use crate::tests::generics::{keyswitch::test_lwe_keyswitch, test_glwe_to_lwe, test_lwe_to_glwe};

#[test]
fn lwe_to_glwe() {
    let log_n: usize = 5;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    test_lwe_to_glwe(&module)
}

#[test]
fn glwe_to_lwe() {
    let log_n: usize = 5;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    test_glwe_to_lwe(&module)
}

#[test]
fn lwe_keyswitch() {
    let log_n: usize = 5;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    test_lwe_keyswitch(&module)
}
