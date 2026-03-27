use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::{api::ModuleNew, layouts::Module};

use crate::leveled::tests::test_suite::encrypt_decrypt::test_encrypt_decrypt;

#[test]
fn encrypt_decrypt() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(32768);
    test_encrypt_decrypt(&module);
}
