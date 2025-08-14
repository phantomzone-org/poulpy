use backend::hal::tests::serialization::test_reader_writer_interface;

use crate::layouts::{
    GGLWEAutomorphismKey, GGLWECiphertext, GGLWETensorKey, GGSWCiphertext, GLWECiphertext, GGLWESwitchingKey,
    GLWEToLWESwitchingKey, LWECiphertext, LWESwitchingKey, LWEToGLWESwitchingKey,
    compressed::{
        GGLWEAutomorphismKeyCompressed, GGLWECiphertextCompressed, GGSWCiphertextCompressed, GLWECiphertextCompressed,
        GGLWESwitchingKeyCompressed, GGLWETensorKeyCompressed, GLWEToLWESwitchingKeyCompressed, LWECiphertextCompressed,
        LWESwitchingKeyCompressed, LWEToGLWESwitchingKeyCompressed,
    },
};

const N_GLWE: usize = 64;
const N_LWE: usize = 32;
const BASEK: usize = 12;
const K: usize = 33;
const ROWS: usize = 2;
const RANK: usize = 2;
const DIGITS: usize = 1;

#[test]
fn glwe_serialization() {
    let original: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(N_GLWE, BASEK, K, RANK);
    backend::hal::tests::serialization::test_reader_writer_interface(original);
}

#[test]
fn glwe_compressed_serialization() {
    let original: GLWECiphertextCompressed<Vec<u8>> = GLWECiphertextCompressed::alloc(N_GLWE, BASEK, K, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_serialization() {
    let original: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(N_LWE, BASEK, K);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_compressed_serialization() {
    let original: LWECiphertextCompressed<Vec<u8>> = LWECiphertextCompressed::alloc(BASEK, K);
    test_reader_writer_interface(original);
}

#[test]
fn test_gglwe_serialization() {
    let original: GGLWECiphertext<Vec<u8>> = GGLWECiphertext::alloc(1024, 12, 54, 3, 1, 2, 2);
    test_reader_writer_interface(original);
}

#[test]
fn test_gglwe_compressed_serialization() {
    let original: GGLWECiphertextCompressed<Vec<u8>> = GGLWECiphertextCompressed::alloc(1024, 12, 54, 3, 1, 2, 2);
    test_reader_writer_interface(original);
}

#[test]
fn test_glwe_switching_key_serialization() {
    let original: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(1024, 12, 54, 3, 1, 2, 2);
    test_reader_writer_interface(original);
}

#[test]
fn test_glwe_switching_key_compressed_serialization() {
    let original: GGLWESwitchingKeyCompressed<Vec<u8>> = GGLWESwitchingKeyCompressed::alloc(1024, 12, 54, 3, 1, 2, 2);
    test_reader_writer_interface(original);
}

#[test]
fn test_automorphism_key_serialization() {
    let original: GGLWEAutomorphismKey<Vec<u8>> = GGLWEAutomorphismKey::alloc(1024, 12, 54, 3, 1, 2);
    test_reader_writer_interface(original);
}

#[test]
fn test_automorphism_key_compressed_serialization() {
    let original: GGLWEAutomorphismKeyCompressed<Vec<u8>> = GGLWEAutomorphismKeyCompressed::alloc(1024, 12, 54, 3, 1, 2);
    test_reader_writer_interface(original);
}

#[test]
fn test_tensor_key_serialization() {
    let original: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(1024, 12, 54, 3, 1, 2);
    test_reader_writer_interface(original);
}

#[test]
fn test_tensor_key_compressed_serialization() {
    let original: GGLWETensorKeyCompressed<Vec<u8>> = GGLWETensorKeyCompressed::alloc(1024, 12, 54, 3, 1, 2);
    test_reader_writer_interface(original);
}

#[test]
fn glwe_to_lwe_switching_key_serialization() {
    let original: GLWEToLWESwitchingKey<Vec<u8>> = GLWEToLWESwitchingKey::alloc(N_GLWE, BASEK, K, ROWS, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn glwe_to_lwe_switching_key_compressed_serialization() {
    let original: GLWEToLWESwitchingKeyCompressed<Vec<u8>> = GLWEToLWESwitchingKeyCompressed::alloc(N_GLWE, BASEK, K, ROWS, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_to_glwe_switching_key_serialization() {
    let original: LWEToGLWESwitchingKey<Vec<u8>> = LWEToGLWESwitchingKey::alloc(N_GLWE, BASEK, K, ROWS, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_to_glwe_switching_key_compressed_serialization() {
    let original: LWEToGLWESwitchingKeyCompressed<Vec<u8>> = LWEToGLWESwitchingKeyCompressed::alloc(N_GLWE, BASEK, K, ROWS, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_switching_key_serialization() {
    let original: LWESwitchingKey<Vec<u8>> = LWESwitchingKey::alloc(N_GLWE, BASEK, K, ROWS);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_switching_key_compressed_serialization() {
    let original: LWESwitchingKeyCompressed<Vec<u8>> = LWESwitchingKeyCompressed::alloc(N_GLWE, BASEK, K, ROWS);
    test_reader_writer_interface(original);
}

#[test]
fn ggsw_serialization() {
    let original: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(N_GLWE, BASEK, K, ROWS, DIGITS, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn ggsw_compressed_serialization() {
    let original: GGSWCiphertextCompressed<Vec<u8>> = GGSWCiphertextCompressed::alloc(N_GLWE, BASEK, K, ROWS, DIGITS, RANK);
    test_reader_writer_interface(original);
}
