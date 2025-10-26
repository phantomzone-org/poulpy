use poulpy_hal::{api::ModuleNew, layouts::Module, test_suite::convolution::test_convolution};

use crate::FFT64Ref;

#[test]
fn test_convolution_fft64_ref() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(8);
    test_convolution(&module);
}
