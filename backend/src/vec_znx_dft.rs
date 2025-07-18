use std::marker::PhantomData;

use crate::znx_base::ZnxInfos;
use crate::{alloc_aligned, Backend, DataView, DataViewMut, VecZnxBig, ZnxMachineWord, ZnxSliceSize, ZnxView, ZnxViewMut, ZnxZero, FFT64, NTT120};
use std::fmt;

pub struct VecZnxDft<D, B: Backend> {
    pub(crate) data: D,
    pub(crate) n: usize,
    pub(crate) cols: usize,
    pub(crate) size: usize,
    pub(crate) _phantom: PhantomData<B>,
}

impl<D, B: Backend> VecZnxDft<D, B> {
    pub fn into_big(self) -> VecZnxBig<D, B> {
        VecZnxBig::<D, B>::from_data(self.data, self.n, self.cols, self.size)
    }
}

impl<D, B: Backend> ZnxInfos for VecZnxDft<D, B> {
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

impl<D> ZnxSliceSize for VecZnxDft<D, FFT64> {
    fn sl(&self) -> usize {
        Self::mw() * self.n() * self.cols()
    }
}

impl<D> ZnxSliceSize for VecZnxDft<D, NTT120> {
    fn sl(&self) -> usize {
        Self::mw() * self.n() * self.cols()
    }
}

impl<D> ZnxMachineWord for VecZnxDft<D, FFT64> {
    fn mw() -> usize {
        1
    }
}

impl<D> ZnxMachineWord for VecZnxDft<D, NTT120> {
    fn mw() -> usize {
        4
    }
}

impl<D, B: Backend> DataView for VecZnxDft<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D, B: Backend> DataViewMut for VecZnxDft<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: AsRef<[u8]>> ZnxView for VecZnxDft<D, FFT64> {
    type Scalar = f64;
}

impl<D: AsRef<[u8]>> ZnxView for VecZnxDft<D, NTT120> {
    type Scalar = i64;
}

impl<D: AsMut<[u8]> + AsRef<[u8]>, B: Backend> VecZnxDft<D, B> where VecZnxDft<D, B>: ZnxMachineWord  {
    pub fn set_size(&mut self, size: usize) {
        assert!(size <= self.max_size());
        self.size = size
    }

    pub fn max_size(&mut self) -> usize {
        self.data.as_ref().len() / (self.n() * Self::mw() * self.cols())
    }
}

pub trait VecZnxDftBytesOf<D, B: Backend> {
    fn bytes_of(n: usize, cols: usize, size: usize) -> usize;
}

impl<D: AsRef<[u8]>, B: Backend> VecZnxDftBytesOf<D, B> for VecZnxDft<D, B> where VecZnxDft<D, B>: ZnxMachineWord {
    fn bytes_of(n: usize, cols: usize, size: usize) -> usize {
        Self::mw() * n * cols * size * size_of::<f64>()
    }
}

impl<D: From<Vec<u8>> + AsRef<[u8]>, B: Backend> VecZnxDft<D, B>
where
    VecZnxDft<D, B>: VecZnxDftBytesOf<D, B>,
{
    pub(crate) fn new(n: usize, cols: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of(n, cols, size));
        Self {
            data: data.into(),
            n: n,
            cols,
            size,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn new_from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(n, cols, size));
        Self {
            data: data.into(),
            n: n,
            cols,
            size,
            _phantom: PhantomData,
        }
    }
}

pub type VecZnxDftOwned<B> = VecZnxDft<Vec<u8>, B>;

impl<D, B: Backend> VecZnxDft<D, B> {
    pub(crate) fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            n,
            cols,
            size,
            _phantom: PhantomData,
        }
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>> ZnxZero for VecZnxDft<D, FFT64>{
    fn zero(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.as_mut_ptr(), 0, self.sl() * self.size());
        }
    }

    fn zero_at(&mut self, i: usize, j: usize) {
        unsafe {
            std::ptr::write_bytes(self.at_mut_ptr(i, j), 0, self.n());
        }
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>> ZnxZero for VecZnxDft<D, NTT120>{
    fn zero(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.as_mut_ptr(), 0, self.sl() * self.size());
        }
    }

    fn zero_at(&mut self, i: usize, j: usize) {
        unsafe {
            std::ptr::write_bytes(self.at_mut_ptr(i, j), 0, 4*self.n());
        }
    }
}

pub trait VecZnxDftToRef<B: Backend> {
    fn to_ref(&self) -> VecZnxDft<&[u8], B>;
}

impl<D: AsRef<[u8]>, B: Backend> VecZnxDftToRef<B> for VecZnxDft<D, B> {
    fn to_ref(&self) -> VecZnxDft<&[u8], B> {
        VecZnxDft {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub trait VecZnxDftToMut<B: Backend> {
    fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B>;
}

impl<D: AsRef<[u8]> + AsMut<[u8]>, B: Backend> VecZnxDftToMut<B> for VecZnxDft<D, B> {
    fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B> {
        VecZnxDft {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D: AsRef<[u8]>> fmt::Display for VecZnxDft<D, FFT64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "VecZnxDft(n={}, cols={}, size={})",
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
