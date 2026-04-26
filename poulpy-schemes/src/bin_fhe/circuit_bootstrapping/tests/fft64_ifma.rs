use poulpy_cpu_ifma::FFT64Ifma;
use poulpy_hal::{api::ModuleNew, layouts::Module};

use crate::bin_fhe::{
    blind_rotation::CGGI,
    circuit_bootstrapping::tests::circuit_bootstrapping::{
        test_circuit_bootstrapping_to_constant, test_circuit_bootstrapping_to_exponent,
    },
};

#[test]
fn to_constant_cggi() {
    let module: Module<FFT64Ifma> = Module::<FFT64Ifma>::new(256);
    test_circuit_bootstrapping_to_constant::<FFT64Ifma, _, CGGI>(&module);
}

#[test]
fn to_exponent_cggi() {
    let module: Module<FFT64Ifma> = Module::<FFT64Ifma>::new(256);
    test_circuit_bootstrapping_to_exponent::<FFT64Ifma, _, CGGI>(&module);
}
