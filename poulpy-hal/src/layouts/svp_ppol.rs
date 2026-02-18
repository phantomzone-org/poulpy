use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::{
    alloc_aligned,
    layouts::{
        Backend, Data, DataMut, DataRef, DataView, DataViewMut, DigestU64, ReaderFrom, WriterTo, ZnxInfos, ZnxSliceSize, ZnxView,
    },
    oep::SvpPPolAllocBytesImpl,
};

#[repr(C)]
#[derive(PartialEq, Eq, Hash)]
pub struct SvpPPol<D: Data, B: Backend> {
    pub data: D,
    pub n: usize,
    pub cols: usize,
    pub _phantom: PhantomData<B>,
}

impl<D: DataRef, B: Backend> DigestU64 for SvpPPol<D, B> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n);
        h.write_usize(self.cols);
        h.finish()
    }
}

impl<D: Data, B: Backend> ZnxSliceSize for SvpPPol<D, B> {
    fn sl(&self) -> usize {
        B::layout_prep_word_count() * self.n()
    }
}

impl<D: DataRef, B: Backend> ZnxView for SvpPPol<D, B> {
    type Scalar = B::ScalarPrep;
}

impl<D: Data, B: Backend> ZnxInfos for SvpPPol<D, B> {
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
        1
    }
}

impl<D: Data, B: Backend> DataView for SvpPPol<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, B: Backend> DataViewMut for SvpPPol<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: Data + From<Vec<u8>>, B: Backend> SvpPPol<D, B>
where
    B: SvpPPolAllocBytesImpl<B>,
{
    pub fn alloc(n: usize, cols: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(B::svp_ppol_bytes_of_impl(n, cols));
        Self {
            data: data.into(),
            n,
            cols,
            _phantom: PhantomData,
        }
    }

    pub fn from_bytes(n: usize, cols: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::svp_ppol_bytes_of_impl(n, cols));
        crate::assert_alignment(data.as_ptr());
        Self {
            data: data.into(),
            n,
            cols,
            _phantom: PhantomData,
        }
    }
}

pub type SvpPPolOwned<B> = SvpPPol<Vec<u8>, B>;

pub trait SvpPPolToRef<B: Backend> {
    fn to_ref(&self) -> SvpPPol<&[u8], B>;
}

impl<D: DataRef, B: Backend> SvpPPolToRef<B> for SvpPPol<D, B> {
    fn to_ref(&self) -> SvpPPol<&[u8], B> {
        SvpPPol {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            _phantom: PhantomData,
        }
    }
}

pub trait SvpPPolToMut<B: Backend> {
    fn to_mut(&mut self) -> SvpPPol<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> SvpPPolToMut<B> for SvpPPol<D, B> {
    fn to_mut(&mut self) -> SvpPPol<&mut [u8], B> {
        SvpPPol {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, B: Backend> SvpPPol<D, B> {
    pub fn from_data(data: D, n: usize, cols: usize) -> Self {
        Self {
            data,
            n,
            cols,
            _phantom: PhantomData,
        }
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut, B: Backend> ReaderFrom for SvpPPol<D, B> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let new_n: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_cols: usize = reader.read_u64::<LittleEndian>()? as usize;
        let len: usize = reader.read_u64::<LittleEndian>()? as usize;

        // SvpPPol is backend-specific so we cannot compute expected_len from metadata alone,
        // but we can at least validate the buffer is large enough before reading.
        let buf: &mut [u8] = self.data.as_mut();
        if buf.len() < len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("SvpPPol buffer too small: self.data.len()={} < read len={len}", buf.len()),
            ));
        }
        reader.read_exact(&mut buf[..len])?;

        // Commit metadata only after successful read
        self.n = new_n;
        self.cols = new_cols;
        Ok(())
    }
}

impl<D: DataRef, B: Backend> WriterTo for SvpPPol<D, B> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.n as u64)?;
        writer.write_u64::<LittleEndian>(self.cols as u64)?;
        let buf: &[u8] = self.data.as_ref();
        writer.write_u64::<LittleEndian>(buf.len() as u64)?;
        writer.write_all(buf)?;
        Ok(())
    }
}

impl<D: DataRef, B: Backend> fmt::Display for SvpPPol<D, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SvpPPol(n={}, cols={})", self.n, self.cols)?;

        for col in 0..self.cols {
            writeln!(f, "Column {col}:")?;
            let coeffs = self.at(col, 0);
            write!(f, "[")?;

            let max_show = 100;
            let show_count = coeffs.len().min(max_show);

            for (i, &coeff) in coeffs.iter().take(show_count).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{coeff}")?;
            }

            if coeffs.len() > max_show {
                write!(f, ", ... ({} more)", coeffs.len() - max_show)?;
            }

            writeln!(f, "]")?;
        }
        Ok(())
    }
}
