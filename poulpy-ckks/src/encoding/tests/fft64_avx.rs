use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{api::ModuleNew, layouts::Module};

use crate::encoding::tests::test_suite::encode_decode::test_encode_decode;

#[test]
fn encode_decode() {
    let module: Module<FFT64Avx> = Module::<FFT64Avx>::new(32768);
    test_encode_decode(&module);
}
