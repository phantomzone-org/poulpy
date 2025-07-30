use backend::{FFT64, Module, ModuleNew};

use crate::gglwe::test_fft64::tensor_key_generic::test_encrypt_sk;

#[test]
fn encrypt_sk() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    (1..4).for_each(|rank| {
        println!("test encrypt_sk rank: {}", rank);
        test_encrypt_sk(&module, 16, 54, 3.2, rank);
    });
}
