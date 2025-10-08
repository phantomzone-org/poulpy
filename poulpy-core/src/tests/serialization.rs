use poulpy_hal::test_suite::serialization::test_reader_writer_interface;

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEAutomorphismKey, GGLWECiphertext, GGLWESwitchingKey, GGLWETensorKey, GGSWCiphertext,
    GLWECiphertext, GLWEToLWEKey, LWECiphertext, LWESwitchingKey, LWEToGLWESwitchingKey, Rank, TorusPrecision,
    compressed::{
        GGLWEAutomorphismKeyCompressed, GGLWECiphertextCompressed, GGLWESwitchingKeyCompressed, GGLWETensorKeyCompressed,
        GGSWCiphertextCompressed, GLWECiphertextCompressed, GLWEToLWESwitchingKeyCompressed, LWECiphertextCompressed,
        LWESwitchingKeyCompressed, LWEToGLWESwitchingKeyCompressed,
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
    let original: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc_with(N_GLWE, BASE2K, K, RANK);
    poulpy_hal::test_suite::serialization::test_reader_writer_interface(original);
}

#[test]
fn glwe_compressed_serialization() {
    let original: GLWECiphertextCompressed<Vec<u8>> = GLWECiphertextCompressed::alloc_with(N_GLWE, BASE2K, K, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_serialization() {
    let original: LWECiphertext<Vec<u8>> = LWECiphertext::alloc_with(N_LWE, BASE2K, K);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_compressed_serialization() {
    let original: LWECiphertextCompressed<Vec<u8>> = LWECiphertextCompressed::alloc_with(BASE2K, K);
    test_reader_writer_interface(original);
}

#[test]
fn test_gglwe_serialization() {
    let original: GGLWECiphertext<Vec<u8>> = GGLWECiphertext::alloc_with(N_GLWE, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_gglwe_compressed_serialization() {
    let original: GGLWECiphertextCompressed<Vec<u8>> =
        GGLWECiphertextCompressed::alloc_with(N_GLWE, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_glwe_switching_key_serialization() {
    let original: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc_with(N_GLWE, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_glwe_switching_key_compressed_serialization() {
    let original: GGLWESwitchingKeyCompressed<Vec<u8>> =
        GGLWESwitchingKeyCompressed::alloc_with(N_GLWE, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_automorphism_key_serialization() {
    let original: GGLWEAutomorphismKey<Vec<u8>> = GGLWEAutomorphismKey::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_automorphism_key_compressed_serialization() {
    let original: GGLWEAutomorphismKeyCompressed<Vec<u8>> =
        GGLWEAutomorphismKeyCompressed::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_tensor_key_serialization() {
    let original: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_tensor_key_compressed_serialization() {
    let original: GGLWETensorKeyCompressed<Vec<u8>> = GGLWETensorKeyCompressed::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn glwe_to_lwe_switching_key_serialization() {
    let original: GLWEToLWEKey<Vec<u8>> = GLWEToLWEKey::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn glwe_to_lwe_switching_key_compressed_serialization() {
    let original: GLWEToLWESwitchingKeyCompressed<Vec<u8>> =
        GLWEToLWESwitchingKeyCompressed::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_to_glwe_switching_key_serialization() {
    let original: LWEToGLWESwitchingKey<Vec<u8>> = LWEToGLWESwitchingKey::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM);
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
    let original: LWESwitchingKey<Vec<u8>> = LWESwitchingKey::alloc_with(N_GLWE, BASE2K, K, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_switching_key_compressed_serialization() {
    let original: LWESwitchingKeyCompressed<Vec<u8>> = LWESwitchingKeyCompressed::alloc_with(N_GLWE, BASE2K, K, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn ggsw_serialization() {
    let original: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn ggsw_compressed_serialization() {
    let original: GGSWCiphertextCompressed<Vec<u8>> = GGSWCiphertextCompressed::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}
