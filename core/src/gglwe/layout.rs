use backend::hal::{
    api::{MatZnxAlloc, MatZnxAllocBytes, VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatPrepare},
    layouts::{Backend, MatZnx, Module, ReaderFrom, Scratch, VmpPMat, WriterTo},
};

use crate::{GLWECiphertext, Infos};

pub trait GGLWEExecLayoutFamily<B: Backend> = VmpPMatAlloc<B> + VmpPMatAllocBytes + VmpPMatPrepare<B>;

pub struct GGLWECiphertext<D> {
    pub(crate) data: MatZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<D: AsRef<[u8]>> GGLWECiphertext<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        GLWECiphertext {
            data: self.data.at(row, col),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> GGLWECiphertext<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext {
            data: self.data.at_mut(row, col),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl GGLWECiphertext<Vec<u8>> {
    pub fn alloc<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid gglwe: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        Self {
            data: module.mat_znx_alloc(rows, rank_in, rank_out + 1, size),
            basek: basek,
            k,
            digits,
        }
    }

    pub fn bytes_of<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize
    where
        Module<B>: MatZnxAllocBytes,
    {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid gglwe: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        module.mat_znx_alloc_bytes(rows, rank_in, rank_out + 1, rows)
    }
}

impl<D> Infos for GGLWECiphertext<D> {
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

impl<D> GGLWECiphertext<D> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.digits
    }

    pub fn rank_in(&self) -> usize {
        self.data.cols_in()
    }

    pub fn rank_out(&self) -> usize {
        self.data.cols_out() - 1
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: AsRef<[u8]> + AsMut<[u8]>> ReaderFrom for GGLWECiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.digits = reader.read_u64::<LittleEndian>()? as usize;
        self.data.read_from(reader)
    }
}

impl<D: AsRef<[u8]>> WriterTo for GGLWECiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        writer.write_u64::<LittleEndian>(self.digits as u64)?;
        self.data.write_to(writer)
    }
}

pub struct GGLWECiphertextExec<D, B: Backend> {
    pub(crate) data: VmpPMat<D, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<B: Backend> GGLWECiphertextExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize, rank_out: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid gglwe: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        Self {
            data: module.vmp_pmat_alloc(rows, rank_in, rank_out + 1, size),
            basek: basek,
            k,
            digits,
        }
    }

    pub fn bytes_of(
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid gglwe: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        module.vmp_pmat_alloc_bytes(rows, rank_in, rank_out + 1, rows)
    }
}

impl<D, B: Backend> Infos for GGLWECiphertextExec<D, B> {
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

impl<D, B: Backend> GGLWECiphertextExec<D, B> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.digits
    }

    pub fn rank_in(&self) -> usize {
        self.data.cols_in()
    }

    pub fn rank_out(&self) -> usize {
        self.data.cols_out() - 1
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>, B: Backend> GGLWECiphertextExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GGLWECiphertext<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: AsRef<[u8]>,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        module.vmp_prepare(&mut self.data, &other.data, scratch);
        self.basek = other.basek;
        self.k = other.k;
        self.digits = other.digits;
    }
}
