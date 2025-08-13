use backend::hal::tests::serialization::test_reader_writer_interface;

use crate::{
    GLWEToLWESwitchingKey, GLWEToLWESwitchingKeyCompressed, LWECiphertext, LWECiphertextCompressed, LWESwitchingKey,
    LWESwitchingKeyCompressed, LWEToGLWESwitchingKey, LWEToGLWESwitchingKeyCompressed,
};

#[test]
fn lwe_serialization() {
    let original: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(771, 12, 54);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_compressed_serialization() {
    let original: LWECiphertextCompressed<Vec<u8>> = LWECiphertextCompressed::alloc(12, 54);
    test_reader_writer_interface(original);
}

#[test]
fn glwe_to_lwe_switching_key_serialization() {
    let original: GLWEToLWESwitchingKey<Vec<u8>> = GLWEToLWESwitchingKey::alloc(1024, 12, 54, 2, 2);
    test_reader_writer_interface(original);
}

#[test]
fn glwe_to_lwe_switching_key_compressed_serialization() {
    let original: GLWEToLWESwitchingKeyCompressed<Vec<u8>> = GLWEToLWESwitchingKeyCompressed::alloc(1024, 12, 54, 2, 2);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_to_glwe_switching_key_serialization() {
    let original: LWEToGLWESwitchingKey<Vec<u8>> = LWEToGLWESwitchingKey::alloc(1024, 12, 54, 2, 2);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_to_glwe_switching_key_compressed_serialization() {
    let original: LWEToGLWESwitchingKeyCompressed<Vec<u8>> = LWEToGLWESwitchingKeyCompressed::alloc(1024, 12, 54, 2, 2);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_switching_key_serialization() {
    let original: LWESwitchingKey<Vec<u8>> = LWESwitchingKey::alloc(1024, 12, 54, 2);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_switching_key_compressed_serialization() {
    let original: LWESwitchingKeyCompressed<Vec<u8>> = LWESwitchingKeyCompressed::alloc(1024, 12, 54, 2);
    test_reader_writer_interface(original);
}
