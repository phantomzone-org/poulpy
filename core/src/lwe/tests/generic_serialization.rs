use backend::hal::tests::serialization::test_reader_writer_interface;

use crate::{LWECiphertext, LWECiphertextCompressed};

#[test]
fn lwe_serialization() {
    let original: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(771, 12, 54);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_serialization_compressed() {
    let original: LWECiphertextCompressed<Vec<u8>> = LWECiphertextCompressed::alloc(12, 54);
    test_reader_writer_interface(original);
}
