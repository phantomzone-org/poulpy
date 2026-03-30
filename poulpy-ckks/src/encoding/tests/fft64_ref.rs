use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::{api::ModuleNew, layouts::Module};

use crate::encoding::tests::test_suite::encode_decode::test_encode_decode;

#[test]
fn encode_decode() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(65536);
    test_encode_decode(&module);
}
