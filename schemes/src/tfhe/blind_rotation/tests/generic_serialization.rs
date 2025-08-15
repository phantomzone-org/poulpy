use backend::hal::tests::serialization::test_reader_writer_interface;

use crate::tfhe::blind_rotation::{BlindRotationKey, BlindRotationKeyCompressed, CGGI};

#[test]
fn test_cggi_blind_rotation_key_serialization() {
    let original: BlindRotationKey<Vec<u8>, CGGI> = BlindRotationKey::alloc(256, 64, 12, 54, 2, 2);
    test_reader_writer_interface(original);
}

#[test]
fn test_cggi_blind_rotation_key_compressed_serialization() {
    let original: BlindRotationKeyCompressed<Vec<u8>, CGGI> = BlindRotationKeyCompressed::alloc(256, 64, 12, 54, 2, 2);
    test_reader_writer_interface(original);
}
