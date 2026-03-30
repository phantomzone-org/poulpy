use crate::encoding::tests::test_suite::encode_decode::test_encode_decode;

#[test]
fn encode_decode() {
    test_encode_decode(65536);
}
