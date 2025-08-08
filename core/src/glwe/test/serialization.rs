use backend::hal::{
    api::VecZnxFillUniform,
    layouts::{Backend, FillUniform, Module, ReaderFrom, WriterTo},
    oep::{ModuleNewImpl, VecZnxAllocImpl, VecZnxFillUniformImpl},
    tests::serialization::test_reader_writer_interface,
};
use sampling::source::Source;

use crate::{GLWECiphertext, GLWECiphertextSeeded, Infos};

pub fn test_glwe_serialization<B: Backend>()
where
    B: ModuleNewImpl<B> + VecZnxAllocImpl<B>,
{
    let module: Module<B> = B::new_impl(1024);
    let ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, 12, 54, 3);
    test_reader_writer_interface(ct);
}

pub fn test_glwe_seeded_serialization<B: Backend>()
where
    B: ModuleNewImpl<B> + VecZnxAllocImpl<B> + VecZnxFillUniformImpl<B>,
{
    let module: Module<B> = B::new_impl(1024);

    let basek: usize = 12;
    let k: usize = 54;
    let rank = 3;

    // Create a random seeded ciphertext
    let mut original_seeded: GLWECiphertextSeeded<Vec<u8>> = GLWECiphertextSeeded::alloc(&module, basek, k, rank);
    original_seeded.seed = [1u8; 32];
    let mut source: Source = Source::new([0u8; 32]);
    original_seeded.data.fill_uniform(&mut source);

    // Write seeded ciphertext ciphertext to a raw Vec<u8> buffer
    let mut buffer: Vec<u8> = Vec::new();
    let write_result: Result<(), std::io::Error> = original_seeded.write_to(&mut buffer);
    assert!(
        write_result.is_ok(),
        "write_to failed with error: {:?}",
        write_result.unwrap_err()
    );

    // Deserialize on the new struct
    let mut reader: &[u8] = &buffer;
    let mut receiver: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k, rank);
    let read_result: Result<(), std::io::Error> = receiver.read_from(&mut reader);
    assert!(
        read_result.is_ok(),
        "read_from failed with error: {:?}",
        read_result.unwrap_err()
    );

    

    // Generates the expected ciphertext
    let mut original: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k, rank);
    let mut source: Source = Source::new(original_seeded.seed);
    (1..original.cols()).for_each(|i| {
        module.vec_znx_fill_uniform(basek, &mut original.data, i, basek * k, &mut source);
    });

    // Check for equality
    assert_eq!(
        &original, &receiver,
        "Deserialized object does not match the original"
    );
}
