use backend::hal::tests::serialization::test_reader_writer_interface;

use crate::{GGSWCiphertext, GGSWCiphertextCompressed};

#[test]
fn ggsw_test_serialization() {
    let original: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(1024, 12, 54, 3, 1, 2);
    test_reader_writer_interface(original);
}

#[test]
fn ggsw_test_compressed_serialization() {
    let original: GGSWCiphertextCompressed<Vec<u8>> = GGSWCiphertextCompressed::alloc(1024, 12, 54, 3, 1, 2);
    test_reader_writer_interface(original);
}
