use backend::hal::tests::serialization::test_reader_writer_interface;

use crate::{
    AutomorphismKey, AutomorphismKeyCompressed, GGLWECiphertext, GGLWECiphertextCompressed, GLWESwitchingKey,
    GLWESwitchingKeyCompressed, GLWETensorKey, GLWETensorKeyCompressed,
};

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
    let original: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(1024, 12, 54, 3, 1, 2, 2);
    test_reader_writer_interface(original);
}

#[test]
fn test_glwe_switching_key_compressed_serialization() {
    let original: GLWESwitchingKeyCompressed<Vec<u8>> = GLWESwitchingKeyCompressed::alloc(1024, 12, 54, 3, 1, 2, 2);
    test_reader_writer_interface(original);
}

#[test]
fn test_automorphism_key_serialization() {
    let original: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(1024, 12, 54, 3, 1, 2);
    test_reader_writer_interface(original);
}

#[test]
fn test_automorphism_key_compressed_serialization() {
    let original: AutomorphismKeyCompressed<Vec<u8>> = AutomorphismKeyCompressed::alloc(1024, 12, 54, 3, 1, 2);
    test_reader_writer_interface(original);
}

#[test]
fn test_tensor_key_serialization() {
    let original: GLWETensorKey<Vec<u8>> = GLWETensorKey::alloc(1024, 12, 54, 3, 1, 2);
    test_reader_writer_interface(original);
}

#[test]
fn test_tensor_key_compressed_serialization() {
    let original: GLWETensorKeyCompressed<Vec<u8>> = GLWETensorKeyCompressed::alloc(1024, 12, 54, 3, 1, 2);
    test_reader_writer_interface(original);
}
