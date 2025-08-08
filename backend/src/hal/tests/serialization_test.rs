use crate::hal::{
    layouts::{MatZnx, ScalarZnx, VecZnx},
    tests::serialization::test_reader_writer_interface,
};

#[test]
fn test_scalar_znx_serialization() {
    let original: ScalarZnx<Vec<u8>> = ScalarZnx::new(1024, 3);
    test_reader_writer_interface(original);
}

#[test]
fn test_vec_znx_serialization() {
    let original: VecZnx<Vec<u8>> = VecZnx::new::<i64>(1024, 3, 4);
    test_reader_writer_interface(original);
}

#[test]
fn test_mat_znx_big_serialization() {
    let original: MatZnx<Vec<u8>> = MatZnx::new(1024, 3, 2, 3, 4);
    test_reader_writer_interface(original);
}
