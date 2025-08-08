use crate::{
    hal::{
        api::ModuleNew,
        layouts::Module,
        tests::vec_znx::{
            test_vec_znx_add_normal, test_vec_znx_encode_vec_i64_hi_norm, test_vec_znx_encode_vec_i64_lo_norm,
            test_vec_znx_fill_uniform,
        },
    },
    implementation::cpu_spqlios::FFT64,
};

#[test]
fn test_vec_znx_fill_uniform_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 12);
    test_vec_znx_fill_uniform(&module);
}

#[test]
fn test_vec_znx_add_normal_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 12);
    test_vec_znx_add_normal(&module);
}

#[test]
fn test_vec_znx_encode_vec_lo_norm_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 8);
    test_vec_znx_encode_vec_i64_lo_norm(&module);
}

#[test]
fn test_vec_znx_encode_vec_hi_norm_fft64() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 8);
    test_vec_znx_encode_vec_i64_hi_norm(&module);
}
