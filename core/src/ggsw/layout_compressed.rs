use backend::hal::{
    api::{MatZnxAlloc, MatZnxAllocBytes, VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, MatZnx, Module, ReaderFrom, WriterTo},
};

use crate::{Decompress, GGLWECiphertextCompressed, GGSWCiphertext, GLWECiphertextCompressed, Infos};

#[derive(PartialEq, Eq)]
pub struct GGSWCiphertextCompressedV1<D: Data> {
    pub(crate) data: GGLWECiphertextCompressed<D>,
}

impl GGSWCiphertextCompressedV1<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        GGSWCiphertextCompressedV1 {
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

impl<D: Data> Infos for GGSWCiphertextCompressedV1<D> {
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

impl<D: Data> GGSWCiphertextCompressedV1<D> {
    pub fn rank(&self) -> usize {
        self.data.rank()
    }

    pub fn digits(&self) -> usize {
        self.data.digits()
    }
}

impl<D: DataMut> ReaderFrom for GGSWCiphertextCompressedV1<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGSWCiphertextCompressedV1<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.data.write_to(writer)
    }
}

impl<D: DataMut, B: Backend, DR: DataRef> Decompress<B, GGSWCiphertextCompressedV1<DR>> for GGSWCiphertext<D> {
    fn decompress(&mut self, module: &Module<B>, other: &GGSWCiphertextCompressedV1<DR>)
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

#[derive(PartialEq, Eq)]
pub struct GGSWCiphertextCompressedV2<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
    pub(crate) seed: Vec<[u8; 32]>,
}

impl GGSWCiphertextCompressedV2<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        let size: usize = k.div_ceil(basek);
        debug_assert!(digits > 0, "invalid ggsw: `digits` == 0");

        debug_assert!(
            size > digits,
            "invalid ggsw: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid ggsw: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        Self {
            data: module.mat_znx_alloc(rows, rank + 1, 1, k.div_ceil(basek)),
            basek,
            k: k,
            digits,
            seed: vec![[0u8; 32]; rows * rank],
        }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize
    where
        Module<B>: MatZnxAllocBytes,
    {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid ggsw: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid ggsw: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        module.mat_znx_alloc_bytes(rows, rank + 1, 1, size)
    }
}

impl<D: DataRef> GGSWCiphertextCompressedV2<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertextCompressed<&[u8]> {
        GLWECiphertextCompressed {
            data: self.data.at(row, col),
            basek: self.basek,
            k: self.k,
            rank: self.rank(),
            seed: self.seed[row * self.cols() + col],
        }
    }
}

impl<D: DataMut> GGSWCiphertextCompressedV2<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertextCompressed<&mut [u8]> {
        let rank: usize = self.rank();
        let cols: usize = self.cols();
        GLWECiphertextCompressed {
            data: self.data.at_mut(row, col),
            basek: self.basek,
            k: self.k,
            rank: rank,
            seed: self.seed[row * cols + col],
        }
    }
}

impl<D: Data> Infos for GGSWCiphertextCompressedV2<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<D: Data> GGSWCiphertextCompressedV2<D> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.digits
    }
}

impl<D: DataMut> ReaderFrom for GGSWCiphertextCompressedV2<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGSWCiphertextCompressedV2<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.data.write_to(writer)
    }
}

impl<D: DataMut, B: Backend, DR: DataRef> Decompress<B, GGSWCiphertextCompressedV2<DR>> for GGSWCiphertext<D> {
    fn decompress(&mut self, module: &Module<B>, other: &GGSWCiphertextCompressedV2<DR>)
    where
        Module<B>: VecZnxFillUniform + VecZnxCopy,
    {
        let rows = self.rows();
        (0..rows).for_each(|row_i| {
            self.at_mut(row_i, 0)
                .decompress(module, &other.at(row_i, 0));
        });
    }
}
