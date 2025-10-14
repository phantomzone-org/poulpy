use poulpy_hal::test_suite::serialization::test_reader_writer_interface;

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
    let original: GLWE<Vec<u8>> = GLWE::alloc(N_GLWE, BASE2K, K, RANK);
    poulpy_hal::test_suite::serialization::test_reader_writer_interface(original);
}

#[test]
fn glwe_compressed_serialization() {
    let original: GLWECompressed<Vec<u8>> = GLWECompressed::alloc(N_GLWE, BASE2K, K, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_serialization() {
    let original: LWE<Vec<u8>> = LWE::alloc(N_LWE, BASE2K, K);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_compressed_serialization() {
    let original: LWECompressed<Vec<u8>> = LWECompressed::alloc(BASE2K, K);
    test_reader_writer_interface(original);
}

#[test]
fn test_gglwe_serialization() {
    let original: GGLWE<Vec<u8>> = GGLWE::alloc(N_GLWE, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_gglwe_compressed_serialization() {
    let original: GGLWECompressed<Vec<u8>> = GGLWECompressed::alloc(N_GLWE, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_glwe_switching_key_serialization() {
    let original: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(N_GLWE, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_glwe_switching_key_compressed_serialization() {
    let original: GLWESwitchingKeyCompressed<Vec<u8>> =
        GLWESwitchingKeyCompressed::alloc(N_GLWE, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_automorphism_key_serialization() {
    let original: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_automorphism_key_compressed_serialization() {
    let original: AutomorphismKeyCompressed<Vec<u8>> = AutomorphismKeyCompressed::alloc(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_tensor_key_serialization() {
    let original: TensorKey<Vec<u8>> = TensorKey::alloc(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_tensor_key_compressed_serialization() {
    let original: TensorKeyCompressed<Vec<u8>> = TensorKeyCompressed::alloc(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn glwe_to_lwe_switching_key_serialization() {
    let original: GLWEToLWESwitchingKey<Vec<u8>> = GLWEToLWESwitchingKey::alloc(N_GLWE, BASE2K, K, RANK, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn glwe_to_lwe_switching_key_compressed_serialization() {
    let original: GLWEToLWESwitchingKeyCompressed<Vec<u8>> =
        GLWEToLWESwitchingKeyCompressed::alloc(N_GLWE, BASE2K, K, RANK, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_to_glwe_switching_key_serialization() {
    let original: LWEToGLWESwitchingKey<Vec<u8>> = LWEToGLWESwitchingKey::alloc(N_GLWE, BASE2K, K, RANK, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_to_glwe_switching_key_compressed_serialization() {
    let original: LWEToGLWESwitchingKeyCompressed<Vec<u8>> =
        LWEToGLWESwitchingKeyCompressed::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_switching_key_serialization() {
    let original: LWESwitchingKey<Vec<u8>> = LWESwitchingKey::alloc(N_GLWE, BASE2K, K, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_switching_key_compressed_serialization() {
    let original: LWESwitchingKeyCompressed<Vec<u8>> = LWESwitchingKeyCompressed::alloc(N_GLWE, BASE2K, K, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn ggsw_serialization() {
    let original: GGSW<Vec<u8>> = GGSW::alloc(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn ggsw_compressed_serialization() {
    let original: GGSWCompressed<Vec<u8>> = GGSWCompressed::alloc(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}
