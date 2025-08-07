use std::fmt;

use crate::{
    alloc_aligned,
    hal::{
        api::{DataView, DataViewMut, ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut, ZnxZero},
        layouts::{Data, DataMut, DataRef, ReaderFrom, WriterTo},
    },
};

#[derive(PartialEq, Eq)]
pub struct VecZnx<D: Data> {
    pub(crate) data: D,
    pub(crate) n: usize,
    pub(crate) cols: usize,
    pub(crate) size: usize,
    pub(crate) max_size: usize,
}

impl<D: DataRef> fmt::Debug for VecZnx<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: Data> ZnxInfos for VecZnx<D> {
    fn cols(&self) -> usize {
        self.cols
    }

    fn rows(&self) -> usize {
        1
    }

    fn n(&self) -> usize {
        self.n
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl<D: Data> ZnxSliceSize for VecZnx<D> {
    fn sl(&self) -> usize {
        self.n() * self.cols()
    }
}

impl<D: Data> DataView for VecZnx<D> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data> DataViewMut for VecZnx<D> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: DataRef> ZnxView for VecZnx<D> {
    type Scalar = i64;
}

impl VecZnx<Vec<u8>> {
    pub fn rsh_scratch_space(n: usize) -> usize {
        n * std::mem::size_of::<i64>()
    }
}

impl<D: DataMut> ZnxZero for VecZnx<D> {
    fn zero(&mut self) {
        self.raw_mut().fill(0)
    }
    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).fill(0);
    }
}

impl<D: DataRef> VecZnx<D> {
    pub fn alloc_bytes<Scalar: Sized>(n: usize, cols: usize, size: usize) -> usize {
        n * cols * size * size_of::<Scalar>()
    }
}

impl<D: DataRef + From<Vec<u8>>> VecZnx<D> {
    pub fn new<Scalar: Sized>(n: usize, cols: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::alloc_bytes::<Scalar>(n, cols, size));
        Self {
            data: data.into(),
            n,
            cols,
            size,
            max_size: size,
        }
    }

    pub fn from_bytes<Scalar: Sized>(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::alloc_bytes::<Scalar>(n, cols, size));
        Self {
            data: data.into(),
            n,
            cols,
            size,
            max_size: size,
        }
    }
}

impl<D: Data> VecZnx<D> {
    pub fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            n,
            cols,
            size,
            max_size: size,
        }
    }
}

impl<D: DataRef> fmt::Display for VecZnx<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "VecZnx(n={}, cols={}, size={})",
            self.n, self.cols, self.size
        )?;

        for col in 0..self.cols {
            writeln!(f, "Column {}:", col)?;
            for size in 0..self.size {
                let coeffs = self.at(col, size);
                write!(f, "  Size {}: [", size)?;

                let max_show = 100;
                let show_count = coeffs.len().min(max_show);

                for (i, &coeff) in coeffs.iter().take(show_count).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", coeff)?;
                }

                if coeffs.len() > max_show {
                    write!(f, ", ... ({} more)", coeffs.len() - max_show)?;
                }

                writeln!(f, "]")?;
            }
        }
        Ok(())
    }
}

pub type VecZnxOwned = VecZnx<Vec<u8>>;
pub type VecZnxMut<'a> = VecZnx<&'a mut [u8]>;
pub type VecZnxRef<'a> = VecZnx<&'a [u8]>;

pub trait VecZnxToRef {
    fn to_ref(&self) -> VecZnx<&[u8]>;
}

impl<D: DataRef> VecZnxToRef for VecZnx<D> {
    fn to_ref(&self) -> VecZnx<&[u8]> {
        VecZnx {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            max_size: self.max_size,
        }
    }
}

pub trait VecZnxToMut {
    fn to_mut(&mut self) -> VecZnx<&mut [u8]>;
}

impl<D: DataMut> VecZnxToMut for VecZnx<D> {
    fn to_mut(&mut self) -> VecZnx<&mut [u8]> {
        VecZnx {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            max_size: self.max_size,
        }
    }
}

impl<D: DataRef> VecZnx<D> {
    pub fn clone(&self) -> VecZnx<Vec<u8>> {
        let self_ref: VecZnx<&[u8]> = self.to_ref();
        VecZnx {
            data: self_ref.data.to_vec(),
            n: self_ref.n,
            cols: self_ref.cols,
            size: self_ref.size,
            max_size: self_ref.max_size,
        }
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut> ReaderFrom for VecZnx<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.n = reader.read_u64::<LittleEndian>()? as usize;
        self.cols = reader.read_u64::<LittleEndian>()? as usize;
        self.size = reader.read_u64::<LittleEndian>()? as usize;
        self.max_size = reader.read_u64::<LittleEndian>()? as usize;
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

impl<D: DataRef> WriterTo for VecZnx<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.n as u64)?;
        writer.write_u64::<LittleEndian>(self.cols as u64)?;
        writer.write_u64::<LittleEndian>(self.size as u64)?;
        writer.write_u64::<LittleEndian>(self.max_size as u64)?;
        let buf: &[u8] = self.data.as_ref();
        writer.write_u64::<LittleEndian>(buf.len() as u64)?;
        writer.write_all(buf)?;
        Ok(())
    }
}
