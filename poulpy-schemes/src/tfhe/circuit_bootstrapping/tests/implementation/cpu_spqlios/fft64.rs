use poulpy_backend::cpu_spqlios::FFT64Spqlios;
use poulpy_hal::{api::ModuleNew, layouts::Module};

use crate::tfhe::{
    blind_rotation::CGGI,
    circuit_bootstrapping::tests::circuit_bootstrapping::{
        test_circuit_bootstrapping_to_constant, test_circuit_bootstrapping_to_exponent,
    },
};

#[test]
fn test_to_constant() {
    let module: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(256);
    test_circuit_bootstrapping_to_constant::<FFT64Spqlios, CGGI>(&module);
}

#[test]
fn test_to_exponent() {
    let module: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(256);
    test_circuit_bootstrapping_to_exponent::<FFT64Spqlios, CGGI>(&module);
}
