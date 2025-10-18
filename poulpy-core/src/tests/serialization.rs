use poulpy_backend::FFT64Ref;
use poulpy_hal::{api::ModuleNew, layouts::Module, test_suite::serialization::test_reader_writer_interface};

use crate::layouts::{
    AutomorphismKey, Base2K, Degree, Dnum, Dsize, GGLWE, GGSW, GLWE, GLWESwitchingKey, GLWEToLWESwitchingKey, LWE,
    LWESwitchingKey, LWEToGLWESwitchingKey, Rank, TensorKey, TorusPrecision,
    compressed::{
        AutomorphismKeyCompressed, GGLWECompressed, GGSWCompressed, GLWECompressed, GLWESwitchingKeyCompressed,
        GLWEToLWESwitchingKeyCompressed, LWECompressed, LWESwitchingKeyCompressed, LWEToGLWESwitchingKeyCompressed,
        TensorKeyCompressed,
    },
};

const N_GLWE: Degree = Degree(64);
const N_LWE: Degree = Degree(32);
const BASE2K: Base2K = Base2K(12);
const K: TorusPrecision = TorusPrecision(33);
const DNUM: Dnum = Dnum(3);
const RANK: Rank = Rank(2);
const DSIZE: Dsize = Dsize(1);

#[test]
fn glwe_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: GLWE<Vec<u8>> = GLWE::alloc(&module, BASE2K, K, RANK);
    poulpy_hal::test_suite::serialization::test_reader_writer_interface(original);
}

#[test]
fn glwe_compressed_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: GLWECompressed<Vec<u8>> = GLWECompressed::alloc(&module, BASE2K, K, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: LWE<Vec<u8>> = LWE::alloc(&module, N_LWE, BASE2K, K);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_compressed_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: LWECompressed<Vec<u8>> = LWECompressed::alloc(&module, BASE2K, K);
    test_reader_writer_interface(original);
}

#[test]
fn test_gglwe_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: GGLWE<Vec<u8>> = GGLWE::alloc(&module, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_gglwe_compressed_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: GGLWECompressed<Vec<u8>> = GGLWECompressed::alloc(&module, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_glwe_switching_key_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(&module, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_glwe_switching_key_compressed_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: GLWESwitchingKeyCompressed<Vec<u8>> =
        GLWESwitchingKeyCompressed::alloc(&module, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_automorphism_key_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_with(&module, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_automorphism_key_compressed_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: AutomorphismKeyCompressed<Vec<u8>> = AutomorphismKeyCompressed::alloc(&module, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_tensor_key_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: TensorKey<Vec<u8>> = TensorKey::alloc(&module, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_tensor_key_compressed_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: TensorKeyCompressed<Vec<u8>> = TensorKeyCompressed::alloc(&module, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn glwe_to_lwe_switching_key_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: GLWEToLWESwitchingKey<Vec<u8>> = GLWEToLWESwitchingKey::alloc(&module, BASE2K, K, RANK, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn glwe_to_lwe_switching_key_compressed_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: GLWEToLWESwitchingKeyCompressed<Vec<u8>> =
        GLWEToLWESwitchingKeyCompressed::alloc(&module, BASE2K, K, RANK, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_to_glwe_switching_key_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: LWEToGLWESwitchingKey<Vec<u8>> = LWEToGLWESwitchingKey::alloc(&module, BASE2K, K, RANK, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_to_glwe_switching_key_compressed_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: LWEToGLWESwitchingKeyCompressed<Vec<u8>> =
        LWEToGLWESwitchingKeyCompressed::alloc_with(&module, BASE2K, K, RANK, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_switching_key_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: LWESwitchingKey<Vec<u8>> = LWESwitchingKey::alloc(&module, BASE2K, K, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_switching_key_compressed_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: LWESwitchingKeyCompressed<Vec<u8>> = LWESwitchingKeyCompressed::alloc(&module, BASE2K, K, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn ggsw_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: GGSW<Vec<u8>> = GGSW::alloc(&module, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn ggsw_compressed_serialization() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(N_GLWE.as_u32() as u64);
    let original: GGSWCompressed<Vec<u8>> = GGSWCompressed::alloc(&module, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}
