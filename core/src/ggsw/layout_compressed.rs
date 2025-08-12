use backend::hal::{
    api::{MatZnxAlloc, MatZnxAllocBytes, VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, MatZnx, Module, ReaderFrom, WriterTo},
};

use crate::{GGLWECiphertextCompressed, GGSWCiphertext, Infos};

#[derive(PartialEq, Eq)]
pub struct GGSWCiphertextCompressed<D: Data> {
    pub(crate) data: GGLWECiphertextCompressed<D>,
}

impl GGSWCiphertextCompressed<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        GGSWCiphertextCompressed {
            data: GGLWECiphertextCompressed::alloc(module, basek, k, rows, digits, rank, rank),
        }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize
    where
        Module<B>: MatZnxAllocBytes,
    {
        GGLWECiphertextCompressed::bytes_of(module, basek, k, rows, digits, rank)
    }
}

impl<D: Data> Infos for GGSWCiphertextCompressed<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        self.data.inner()
    }

    fn basek(&self) -> usize {
        self.data.basek()
    }

    fn k(&self) -> usize {
        self.data.k()
    }
}

impl<D: Data> GGSWCiphertextCompressed<D> {
    pub fn rank(&self) -> usize {
        self.data.rank()
    }

    pub fn digits(&self) -> usize {
        self.data.digits()
    }
}

impl<D: DataMut> ReaderFrom for GGSWCiphertextCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGSWCiphertextCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.data.write_to(writer)
    }
}

impl<D: DataMut> GGSWCiphertext<D> {
    pub fn decompress<DataOther: DataRef, B: Backend>(&mut self, module: &Module<B>, other: &GGSWCiphertextCompressed<DataOther>)
    where
        Module<B>: VecZnxFillUniform + VecZnxCopy,
    {
        let rows = self.rows();
        (0..rows).for_each(|row_i| {
            self.at_mut(row_i, 0)
                .decompress(module, &other.data.at(row_i, 0));
        });
    }
}
