use std::fmt::Debug;

use sampling::source::Source;

use crate::hal::{
    api::ZnxZero,
    layouts::{Backend, FillUniform, ReaderFrom, WriterTo},
};

/// Generic test for serialization.
pub fn test_reader_writer_interface<T, B: Backend>(mut original: T)
where
    T: WriterTo<B> + ReaderFrom<B> + PartialEq + Eq + Debug + Clone + ZnxZero + FillUniform,
{
    let mut source: Source = Source::new([0u8; 32]);
    original.fill_uniform(&mut source);

    // Write to a raw Vec<u8> buffer
    let mut buffer: Vec<u8> = Vec::new();
    let write_result: Result<(), std::io::Error> = original.write_to(&mut buffer);
    assert!(
        write_result.is_ok(),
        "write_to failed with error: {:?}",
        write_result.unwrap_err()
    );

    // Read from a byte slice reference
    let mut reader: &[u8] = &buffer;

    // Clones the original struct, and zeroes it
    let mut receiver: T = original.clone();
    receiver.zero();

    // Deserialize on the new struct
    let read_result: Result<(), std::io::Error> = receiver.read_from(&mut reader);
    assert!(
        read_result.is_ok(),
        "read_from failed with error: {:?}",
        read_result.unwrap_err()
    );

    // Check for equality
    assert_eq!(
        &original, &receiver,
        "Deserialized object does not match the original"
    );
}
