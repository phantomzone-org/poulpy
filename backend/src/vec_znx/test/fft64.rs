use crate::{
    FFT64, Module, ModuleNew,
    test::generics::{vec_znx_add_normal, vec_znx_encode_vec_i64_hi_norm, vec_znx_encode_vec_i64_lo_norm, vec_znx_fill_uniform},
};

#[test]
fn test_vec_znx_fill_uniform() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 12);
    vec_znx_fill_uniform(&module);
}

#[test]
fn test_vec_znx_add_normal() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 12);
    vec_znx_add_normal(&module);
}

#[test]
fn test_vec_znx_encode_vec_lo_norm() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 8);
    vec_znx_encode_vec_i64_lo_norm(&module);
}

#[test]
fn test_vec_znx_encode_vec_hi_norm() {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << 8);
    vec_znx_encode_vec_i64_hi_norm(&module);
}
