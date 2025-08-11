use backend::hal::{
    api::VecZnxAlloc,
    layouts::{Backend, Module},
    tests::serialization::test_reader_writer_interface,
};

use crate::{GLWECiphertext, GLWECiphertextCompressed};

pub(crate) fn test_serialization<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxAlloc,
{
    let original: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, 12, 54, 3);
    test_reader_writer_interface(original);
}

pub(crate) fn test_serialization_compressed<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxAlloc,
{
    let original: GLWECiphertextCompressed<Vec<u8>> = GLWECiphertextCompressed::alloc(module, 12, 54, 3);
    test_reader_writer_interface(original);
}
