use backend::hal::{
    api::{MatZnxAlloc, MatZnxAllocBytes, VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, MatZnx, Module, ReaderFrom, WriterTo},
};

use crate::{Decompress, GGSWCiphertext, GLWECiphertextCompressed, Infos};

#[derive(PartialEq, Eq)]
pub struct GGSWCiphertextCompressed<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
    pub(crate) rank: usize,
    pub(crate) seed: Vec<[u8; 32]>,
}

impl GGSWCiphertextCompressed<Vec<u8>> {
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
            rank,
            seed: vec![[0u8; 32]; rows * (rank + 1)],
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

impl<D: DataRef> GGSWCiphertextCompressed<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertextCompressed<&[u8]> {
        GLWECiphertextCompressed {
            data: self.data.at(row, col),
            basek: self.basek,
            k: self.k,
            rank: self.rank(),
            seed: self.seed[row * (self.rank() + 1) + col],
        }
    }
}

impl<D: DataMut> GGSWCiphertextCompressed<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertextCompressed<&mut [u8]> {
        let rank: usize = self.rank();
        GLWECiphertextCompressed {
            data: self.data.at_mut(row, col),
            basek: self.basek,
            k: self.k,
            rank: rank,
            seed: self.seed[row * (rank + 1) + col],
        }
    }
}

impl<D: Data> Infos for GGSWCiphertextCompressed<D> {
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

impl<D: Data> GGSWCiphertextCompressed<D> {
    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn digits(&self) -> usize {
        self.digits
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

impl<D: DataMut, B: Backend, DR: DataRef> Decompress<B, GGSWCiphertextCompressed<DR>> for GGSWCiphertext<D> {
    fn decompress(&mut self, module: &Module<B>, other: &GGSWCiphertextCompressed<DR>)
    where
        Module<B>: VecZnxFillUniform + VecZnxCopy,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), other.rank())
        }

        let rows: usize = self.rows();
        let rank: usize = self.rank();
        (0..rows).for_each(|row_i| {
            (0..rank + 1).for_each(|col_j| {
                self.at_mut(row_i, col_j)
                    .decompress(module, &other.at(row_i, col_j));
            });
        });
    }
}
