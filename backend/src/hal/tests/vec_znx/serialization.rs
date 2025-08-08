use std::fmt::Debug;

use crate::hal::layouts::{ReaderFrom, VecZnx, WriterTo};


/// Generic test for any type that implements ReaderFrom, WriterTo, PartialEq, and Eq.
///
/// - `original`: the original object to serialize
/// - `constructor`: a closure that returns a new empty instance for deserialization
pub fn test_reader_writer_interface<T, F>(original: &T, mut constructor: F)
where
    T: WriterTo + ReaderFrom + PartialEq + Eq + Debug,
    F: FnMut() -> T,
{
    // Write to a raw Vec<u8> buffer
    let mut buffer = Vec::new();
    let write_result = original.write_to(&mut buffer);
    assert!(
        write_result.is_ok(),
        "write_to failed with error: {:?}",
        write_result.unwrap_err()
    );

    // Read from a byte slice reference
    let mut reader: &[u8] = &buffer;
    let mut deserialized = constructor();
    let read_result = deserialized.read_from(&mut reader);
    assert!(
        read_result.is_ok(),
        "read_from failed with error: {:?}",
        read_result.unwrap_err()
    );

    // Check for equality
    assert_eq!(
        original, &deserialized,
        "Deserialized object does not match the original"
    );
}

#[test]
fn test_vec_znx_serialization(){
    let (n, cols, size) = (1024, 3, 4);
    let original: VecZnx<Vec<u8>> = VecZnx::new::<i64>(n, cols, size);
    let constructor = || VecZnx::new::<i64>(n, cols, size);
    test_reader_writer_interface(&original, &constructor);
    
}