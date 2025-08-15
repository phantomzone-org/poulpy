use backend::{
    hal::{api::ModuleNew, layouts::Module},
    implementation::cpu_spqlios::FFT64,
};

use crate::tfhe::{
    blind_rotation::CGGI,
    circuit_bootstrapping::tests::circuit_bootstrapping::{
        test_circuit_bootstrapping_to_constant, test_circuit_bootstrapping_to_exponent,
    },
};

#[test]
fn test_to_constant() {
    let module: Module<FFT64> = Module::<FFT64>::new(256);
    test_circuit_bootstrapping_to_constant::<FFT64, CGGI>(&module);
}

#[test]
fn test_to_exponent() {
    let module: Module<FFT64> = Module::<FFT64>::new(256);
    test_circuit_bootstrapping_to_exponent::<FFT64, CGGI>(&module);
}
