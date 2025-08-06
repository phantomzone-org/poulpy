use backend::hal::{
    api::{MatZnxAlloc, MatZnxAllocBytes, VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatPrepare},
    layouts::{Backend, Data, DataMut, DataRef, MatZnx, Module, ReaderFrom, Scratch, VmpPMat, WriterTo},
};

use crate::{GLWECiphertext, Infos};

pub trait GGSWLayoutFamily<B: Backend> = VmpPMatAlloc<B> + VmpPMatAllocBytes + VmpPMatPrepare<B>;

#[derive(PartialEq, Eq)]
pub struct GGSWCiphertext<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<D: DataRef> GGSWCiphertext<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        GLWECiphertext {
            data: self.data.at(row, col),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl<D: DataMut> GGSWCiphertext<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext {
            data: self.data.at_mut(row, col),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl GGSWCiphertext<Vec<u8>> {
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
            data: module.mat_znx_alloc(rows, rank + 1, rank + 1, k.div_ceil(basek)),
            basek,
            k: k,
            digits,
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

        module.mat_znx_alloc_bytes(rows, rank + 1, rank + 1, size)
    }
}

impl<D: Data> Infos for GGSWCiphertext<D> {
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

impl<D: Data> GGSWCiphertext<D> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.digits
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut> ReaderFrom for GGSWCiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.digits = reader.read_u64::<LittleEndian>()? as usize;
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGSWCiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        writer.write_u64::<LittleEndian>(self.digits as u64)?;
        self.data.write_to(writer)
    }
}

#[derive(PartialEq, Eq)]
pub struct GGSWCiphertextExec<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<B: Backend> GGSWCiphertextExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: GGSWLayoutFamily<B>,
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
            data: module.vmp_pmat_alloc(rows, rank + 1, rank + 1, k.div_ceil(basek)),
            basek,
            k: k,
            digits,
        }
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize
    where
        Module<B>: GGSWLayoutFamily<B>,
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

        module.vmp_pmat_alloc_bytes(rows, rank + 1, rank + 1, size)
    }

    pub fn from<DataOther: DataRef>(
        module: &Module<B>,
        other: &GGSWCiphertext<DataOther>,
        scratch: &mut Scratch<B>,
    ) -> GGSWCiphertextExec<Vec<u8>, B>
    where
        Module<B>: GGSWLayoutFamily<B>,
    {
        let mut ggsw_exec: GGSWCiphertextExec<Vec<u8>, B> = Self::alloc(
            module,
            other.basek(),
            other.k(),
            other.rows(),
            other.digits(),
            other.rank(),
        );
        ggsw_exec.prepare(module, other, scratch);
        ggsw_exec
    }
}

impl<D: Data, B: Backend> Infos for GGSWCiphertextExec<D, B> {
    type Inner = VmpPMat<D, B>;

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

impl<D: Data, B: Backend> GGSWCiphertextExec<D, B> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.digits
    }
}

impl<DataSelf: DataMut, B: Backend> GGSWCiphertextExec<DataSelf, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GGSWCiphertext<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: GGSWLayoutFamily<B>,
    {
        module.vmp_prepare(&mut self.data, &other.data, scratch);
        self.k = other.k;
        self.basek = other.basek;
        self.digits = other.digits;
    }
}
