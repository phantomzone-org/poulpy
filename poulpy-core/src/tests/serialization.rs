use poulpy_hal::test_suite::serialization::test_reader_writer_interface;

use crate::layouts::{
    AutomorphismKey, Base2K, Degree, Dnum, Dsize, GGLWE, GGSW, GLWECiphertext, GLWESwitchingKey, GLWEToLWESwitchingKey,
    LWECiphertext, LWESwitchingKey, LWEToGLWESwitchingKey, Rank, TensorKey, TorusPrecision,
    compressed::{
        AutomorphismKeyCompressed, GGLWECiphertextCompressed, GGSWCiphertextCompressed, GLWECiphertextCompressed,
        GLWESwitchingKeyCompressed, GLWEToLWESwitchingKeyCompressed, LWECiphertextCompressed, LWESwitchingKeyCompressed,
        LWEToGLWESwitchingKeyCompressed, TensorKeyCompressed,
    },
};

const N_GLWE: Degree = Degree(64);
const N_LWE: Degree = Degree(32);
const BASE2K: Base2K = Base2K(12);
const K: TorusPrecision = TorusPrecision(33);
const DNUM: Dnum = Dnum(3);
const RANK: Rank = Rank(2);
const DSIZE: Dsize = Dsize(1);

#[test]
fn glwe_serialization() {
    let original: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc_with(N_GLWE, BASE2K, K, RANK);
    poulpy_hal::test_suite::serialization::test_reader_writer_interface(original);
}

#[test]
fn glwe_compressed_serialization() {
    let original: GLWECiphertextCompressed<Vec<u8>> = GLWECiphertextCompressed::alloc_with(N_GLWE, BASE2K, K, RANK);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_serialization() {
    let original: LWECiphertext<Vec<u8>> = LWECiphertext::alloc_with(N_LWE, BASE2K, K);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_compressed_serialization() {
    let original: LWECiphertextCompressed<Vec<u8>> = LWECiphertextCompressed::alloc_with(BASE2K, K);
    test_reader_writer_interface(original);
}

#[test]
fn test_gglwe_serialization() {
    let original: GGLWE<Vec<u8>> = GGLWE::alloc_with(N_GLWE, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_gglwe_compressed_serialization() {
    let original: GGLWECiphertextCompressed<Vec<u8>> =
        GGLWECiphertextCompressed::alloc_with(N_GLWE, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_glwe_switching_key_serialization() {
    let original: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_with(N_GLWE, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_glwe_switching_key_compressed_serialization() {
    let original: GLWESwitchingKeyCompressed<Vec<u8>> =
        GLWESwitchingKeyCompressed::alloc_with(N_GLWE, BASE2K, K, RANK, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_automorphism_key_serialization() {
    let original: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_automorphism_key_compressed_serialization() {
    let original: AutomorphismKeyCompressed<Vec<u8>> =
        AutomorphismKeyCompressed::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_tensor_key_serialization() {
    let original: TensorKey<Vec<u8>> = TensorKey::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn test_tensor_key_compressed_serialization() {
    let original: TensorKeyCompressed<Vec<u8>> = TensorKeyCompressed::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn glwe_to_lwe_switching_key_serialization() {
    let original: GLWEToLWESwitchingKey<Vec<u8>> = GLWEToLWESwitchingKey::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn glwe_to_lwe_switching_key_compressed_serialization() {
    let original: GLWEToLWESwitchingKeyCompressed<Vec<u8>> =
        GLWEToLWESwitchingKeyCompressed::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_to_glwe_switching_key_serialization() {
    let original: LWEToGLWESwitchingKey<Vec<u8>> = LWEToGLWESwitchingKey::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_to_glwe_switching_key_compressed_serialization() {
    let original: LWEToGLWESwitchingKeyCompressed<Vec<u8>> =
        LWEToGLWESwitchingKeyCompressed::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_switching_key_serialization() {
    let original: LWESwitchingKey<Vec<u8>> = LWESwitchingKey::alloc_with(N_GLWE, BASE2K, K, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn lwe_switching_key_compressed_serialization() {
    let original: LWESwitchingKeyCompressed<Vec<u8>> = LWESwitchingKeyCompressed::alloc_with(N_GLWE, BASE2K, K, DNUM);
    test_reader_writer_interface(original);
}

#[test]
fn ggsw_serialization() {
    let original: GGSW<Vec<u8>> = GGSW::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}

#[test]
fn ggsw_compressed_serialization() {
    let original: GGSWCiphertextCompressed<Vec<u8>> = GGSWCiphertextCompressed::alloc_with(N_GLWE, BASE2K, K, RANK, DNUM, DSIZE);
    test_reader_writer_interface(original);
}
