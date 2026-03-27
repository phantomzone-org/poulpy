use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{api::ModuleNew, layouts::Module};

use crate::leveled::tests::test_suite::encrypt_decrypt::test_encrypt_decrypt;

#[test]
fn encrypt_decrypt() {
    let module: Module<FFT64Avx> = Module::<FFT64Avx>::new(32768);
    test_encrypt_decrypt(&module);
}
