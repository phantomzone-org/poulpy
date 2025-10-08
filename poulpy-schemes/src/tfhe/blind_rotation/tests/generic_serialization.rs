use poulpy_hal::test_suite::serialization::test_reader_writer_interface;

use crate::tfhe::blind_rotation::{
    BlindRotationKey, BlindRotationKeyAlloc, BlindRotationKeyCompressed, BlindRotationKeyLayout, CGGI,
};

#[test]
fn test_cggi_blind_rotation_key_serialization() {
    let layout: BlindRotationKeyLayout = BlindRotationKeyLayout {
        n_glwe: 256_u32.into(),
        n_lwe: 64_usize.into(),
        base2k: 12_usize.into(),
        k: 54_usize.into(),
        dnum: 2_usize.into(),
        rank: 2_usize.into(),
    };

    let original: BlindRotationKey<Vec<u8>, CGGI> = BlindRotationKey::alloc(&layout);
    test_reader_writer_interface(original);
}

#[test]
fn test_cggi_blind_rotation_key_compressed_serialization() {
    let layout: BlindRotationKeyLayout = BlindRotationKeyLayout {
        n_glwe: 256_u32.into(),
        n_lwe: 64_usize.into(),
        base2k: 12_usize.into(),
        k: 54_usize.into(),
        dnum: 2_usize.into(),
        rank: 2_usize.into(),
    };

    let original: BlindRotationKeyCompressed<Vec<u8>, CGGI> = BlindRotationKeyCompressed::alloc(&layout);
    test_reader_writer_interface(original);
}
