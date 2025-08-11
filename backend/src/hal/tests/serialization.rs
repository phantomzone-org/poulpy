use std::fmt::Debug;

use sampling::source::Source;

use crate::hal::{
    api::{FillUniform, ZnxZero},
    layouts::{ReaderFrom, WriterTo},
};

/// Generic test for serialization and deserialization.
///
/// - `T` must implement I/O traits, zeroing, cloning, and random filling.
pub fn test_reader_writer_interface<T>(mut original: T)
where
    T: WriterTo + ReaderFrom + PartialEq + Eq + Debug + Clone + ZnxZero + FillUniform,
{
    // Fill original with uniform random data
    let mut source = Source::new([0u8; 32]);
    original.fill_uniform(&mut source);

    // Serialize into a buffer
    let mut buffer = Vec::new();
    original.write_to(&mut buffer).expect("write_to failed");

    // Prepare receiver: same shape, but zeroed
    let mut receiver = original.clone();
    receiver.zero();

    // Deserialize from buffer
    let mut reader: &[u8] = &buffer;
    receiver.read_from(&mut reader).expect("read_from failed");

    // Ensure serialization round-trip correctness
    assert_eq!(
        &original, &receiver,
        "Deserialized object does not match the original"
    );
}

#[test]
fn scalar_znx_serialize() {
    let original: crate::hal::layouts::ScalarZnx<Vec<u8>> = crate::hal::layouts::ScalarZnx::alloc(1024, 3);
    test_reader_writer_interface(original);
}

#[test]
fn vec_znx_serialize() {
    let original: crate::hal::layouts::VecZnx<Vec<u8>> = crate::hal::layouts::VecZnx::alloc::<i64>(1024, 3, 4);
    test_reader_writer_interface(original);
}

#[test]
fn mat_znx_serialize() {
    let original: crate::hal::layouts::MatZnx<Vec<u8>> = crate::hal::layouts::MatZnx::alloc(1024, 3, 2, 2, 4);
    test_reader_writer_interface(original);
}
