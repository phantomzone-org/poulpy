use backend::hal::tests::serialization::test_reader_writer_interface;

use crate::tfhe::blind_rotation::{BlindRotationKeyCGGI, BlindRotationKeyCGGICompressed};

#[test]
fn test_cggi_blind_rotation_key_serialization() {
    let original: BlindRotationKeyCGGI<Vec<u8>> = BlindRotationKeyCGGI::alloc(256, 64, 12, 54, 2, 2);
    test_reader_writer_interface(original);
}

#[test]
fn test_cggi_blind_rotation_key_compressed_serialization() {
    let original: BlindRotationKeyCGGICompressed<Vec<u8>> = BlindRotationKeyCGGICompressed::alloc(256, 64, 12, 54, 2, 2);
    test_reader_writer_interface(original);
}
