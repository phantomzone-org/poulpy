use crate::{
    alloc_aligned,
    api::{DataView, DataViewMut, FillUniform, Reset, ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut, ZnxZero},
    layouts::{Data, DataMut, DataRef, ReaderFrom, ToOwnedDeep, VecZnx, WriterTo},
    source::Source,
};
use std::fmt;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rand::RngCore;

#[derive(PartialEq, Eq, Clone)]
pub struct MatZnx<D: Data> {
    data: D,
    n: usize,
    size: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
}

impl<D: DataRef> ToOwnedDeep for MatZnx<D> {
    type Owned = MatZnx<Vec<u8>>;
    fn to_owned_deep(&self) -> Self::Owned {
        MatZnx {
            data: self.data.as_ref().to_vec(),
            n: self.n,
            size: self.size,
            rows: self.rows,
            cols_in: self.cols_in,
            cols_out: self.cols_out,
        }
    }
}

impl<D: DataRef> fmt::Debug for MatZnx<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: Data> ZnxInfos for MatZnx<D> {
    fn cols(&self) -> usize {
        self.cols_in
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn n(&self) -> usize {
        self.n
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl<D: Data> ZnxSliceSize for MatZnx<D> {
    fn sl(&self) -> usize {
        self.n() * self.cols_out()
    }
}

impl<D: Data> DataView for MatZnx<D> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data> DataViewMut for MatZnx<D> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: DataRef> ZnxView for MatZnx<D> {
    type Scalar = i64;
}

impl<D: Data> MatZnx<D> {
    pub fn cols_in(&self) -> usize {
        self.cols_in
    }

    pub fn cols_out(&self) -> usize {
        self.cols_out
    }
}

impl MatZnx<Vec<u8>> {
    pub fn alloc_bytes(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        rows * cols_in * VecZnx::<Vec<u8>>::alloc_bytes(n, cols_out, size)
    }

    pub fn alloc(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned(Self::alloc_bytes(n, rows, cols_in, cols_out, size));
        Self {
            data,
            n,
            size,
            rows,
            cols_in,
            cols_out,
        }
    }

    pub fn from_bytes(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::alloc_bytes(n, rows, cols_in, cols_out, size));
        Self {
            data,
            n,
            size,
            rows,
            cols_in,
            cols_out,
        }
    }
}

impl<D: DataRef> MatZnx<D> {
    pub fn at(&self, row: usize, col: usize) -> VecZnx<&[u8]> {
        #[cfg(debug_assertions)]
        {
            assert!(row < self.rows(), "rows: {} >= {}", row, self.rows());
            assert!(col < self.cols_in(), "cols: {} >= {}", col, self.cols_in());
        }

        let self_ref: MatZnx<&[u8]> = self.to_ref();
        let nb_bytes: usize = VecZnx::<Vec<u8>>::alloc_bytes(self.n, self.cols_out, self.size);
        let start: usize = nb_bytes * self.cols() * row + col * nb_bytes;
        let end: usize = start + nb_bytes;

        VecZnx {
            data: &self_ref.data[start..end],
            n: self.n,
            cols: self.cols_out,
            size: self.size,
            max_size: self.size,
        }
    }
}

impl<D: DataMut> MatZnx<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> VecZnx<&mut [u8]> {
        #[cfg(debug_assertions)]
        {
            assert!(row < self.rows(), "rows: {} >= {}", row, self.rows());
            assert!(col < self.cols_in(), "cols: {} >= {}", col, self.cols_in());
        }

        let n: usize = self.n();
        let cols_out: usize = self.cols_out();
        let cols_in: usize = self.cols_in();
        let size: usize = self.size();

        let self_ref: MatZnx<&mut [u8]> = self.to_mut();
        let nb_bytes: usize = VecZnx::<Vec<u8>>::alloc_bytes(n, cols_out, size);
        let start: usize = nb_bytes * cols_in * row + col * nb_bytes;
        let end: usize = start + nb_bytes;

        VecZnx {
            data: &mut self_ref.data[start..end],
            n,
            cols: cols_out,
            size,
            max_size: size,
        }
    }
}

impl<D: DataMut> FillUniform for MatZnx<D> {
    fn fill_uniform(&mut self, source: &mut Source) {
        source.fill_bytes(self.data.as_mut());
    }
}

impl<D: DataMut> Reset for MatZnx<D> {
    fn reset(&mut self) {
        self.zero();
        self.n = 0;
        self.size = 0;
        self.rows = 0;
        self.cols_in = 0;
        self.cols_out = 0;
    }
}

pub type MatZnxOwned = MatZnx<Vec<u8>>;
pub type MatZnxMut<'a> = MatZnx<&'a mut [u8]>;
pub type MatZnxRef<'a> = MatZnx<&'a [u8]>;

pub trait MatZnxToRef {
    fn to_ref(&self) -> MatZnx<&[u8]>;
}

impl<D: DataRef> MatZnxToRef for MatZnx<D> {
    fn to_ref(&self) -> MatZnx<&[u8]> {
        MatZnx {
            data: self.data.as_ref(),
            n: self.n,
            rows: self.rows,
            cols_in: self.cols_in,
            cols_out: self.cols_out,
            size: self.size,
        }
    }
}

pub trait MatZnxToMut {
    fn to_mut(&mut self) -> MatZnx<&mut [u8]>;
}

impl<D: DataMut> MatZnxToMut for MatZnx<D> {
    fn to_mut(&mut self) -> MatZnx<&mut [u8]> {
        MatZnx {
            data: self.data.as_mut(),
            n: self.n,
            rows: self.rows,
            cols_in: self.cols_in,
            cols_out: self.cols_out,
            size: self.size,
        }
    }
}

impl<D: Data> MatZnx<D> {
    pub fn from_data(data: D, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        Self {
            data,
            n,
            rows,
            cols_in,
            cols_out,
            size,
        }
    }
}

impl<D: DataMut> ReaderFrom for MatZnx<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.n = reader.read_u64::<LittleEndian>()? as usize;
        self.size = reader.read_u64::<LittleEndian>()? as usize;
        self.rows = reader.read_u64::<LittleEndian>()? as usize;
        self.cols_in = reader.read_u64::<LittleEndian>()? as usize;
        self.cols_out = reader.read_u64::<LittleEndian>()? as usize;
        let len: usize = reader.read_u64::<LittleEndian>()? as usize;
        let buf: &mut [u8] = self.data.as_mut();
        if buf.len() != len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("self.data.len()={} != read len={}", buf.len(), len),
            ));
        }
        reader.read_exact(&mut buf[..len])?;
        Ok(())
    }
}

impl<D: DataRef> WriterTo for MatZnx<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.n as u64)?;
        writer.write_u64::<LittleEndian>(self.size as u64)?;
        writer.write_u64::<LittleEndian>(self.rows as u64)?;
        writer.write_u64::<LittleEndian>(self.cols_in as u64)?;
        writer.write_u64::<LittleEndian>(self.cols_out as u64)?;
        let buf: &[u8] = self.data.as_ref();
        writer.write_u64::<LittleEndian>(buf.len() as u64)?;
        writer.write_all(buf)?;
        Ok(())
    }
}

impl<D: DataRef> fmt::Display for MatZnx<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "MatZnx(n={}, rows={}, cols_in={}, cols_out={}, size={})",
            self.n, self.rows, self.cols_in, self.cols_out, self.size
        )?;

        for row_i in 0..self.rows {
            writeln!(f, "Row {}:", row_i)?;
            for col_i in 0..self.cols_in {
                writeln!(f, "cols_in {}:", col_i)?;
                writeln!(f, "{}:", self.at(row_i, col_i))?;
            }
        }
        Ok(())
    }
}

impl<D: DataMut> ZnxZero for MatZnx<D> {
    fn zero(&mut self) {
        self.raw_mut().fill(0)
    }

    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).zero();
    }
}
