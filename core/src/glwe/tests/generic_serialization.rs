use backend::hal::tests::serialization::test_reader_writer_interface;

use crate::{GLWECiphertext, GLWECiphertextCompressed};

#[test]
fn test_serialization() {
    let original: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(1024, 12, 54, 3);
    test_reader_writer_interface(original);
}

#[test]
fn test_serialization_compressed() {
    let original: GLWECiphertextCompressed<Vec<u8>> = GLWECiphertextCompressed::alloc(1024, 12, 54, 3);
    test_reader_writer_interface(original);
}
