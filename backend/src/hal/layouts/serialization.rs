use std::io::{Read, Result, Write};

use crate::hal::layouts::Backend;

pub trait WriterTo<B: Backend> {
    fn write_to<W: Write>(&self, writer: &mut W) -> Result<()>;
}

pub trait ReaderFrom<B: Backend> {
    fn read_from<R: Read>(&mut self, reader: &mut R) -> Result<()>;
}
