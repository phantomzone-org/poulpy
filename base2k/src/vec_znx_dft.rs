use std::marker::PhantomData;

use crate::ffi::vec_znx_dft;
use crate::znx_base::ZnxInfos;
use crate::{Backend, DataView, DataViewMut, FFT64, Module, ZnxSliceSize, ZnxView, alloc_aligned};
use std::fmt;

pub struct VecZnxDft<D, B: Backend> {
    data: D,
    n: usize,
    cols: usize,
    size: usize,
    _phantom: PhantomData<B>,
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
        self.n() * self.cols()
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

pub(crate) fn bytes_of_vec_znx_dft<B: Backend>(module: &Module<B>, cols: usize, size: usize) -> usize {
    unsafe { vec_znx_dft::bytes_of_vec_znx_dft(module.ptr, size as u64) as usize * cols }
}

impl<D: From<Vec<u8>>, B: Backend> VecZnxDft<D, B> {
    pub(crate) fn new(module: &Module<B>, cols: usize, size: usize) -> Self {
        let data = alloc_aligned::<u8>(bytes_of_vec_znx_dft(module, cols, size));
        Self {
            data: data.into(),
            n: module.n(),
            cols,
            size,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn new_from_bytes(module: &Module<B>, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == bytes_of_vec_znx_dft(module, cols, size));
        Self {
            data: data.into(),
            n: module.n(),
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

pub trait VecZnxDftToRef<B: Backend> {
    fn to_ref(&self) -> VecZnxDft<&[u8], B>;
}

pub trait VecZnxDftToMut<B: Backend> {
    fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B>;
}

impl<B: Backend> VecZnxDftToMut<B> for VecZnxDft<Vec<u8>, B> {
    fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B> {
        VecZnxDft {
            data: self.data.as_mut_slice(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend> VecZnxDftToRef<B> for VecZnxDft<Vec<u8>, B> {
    fn to_ref(&self) -> VecZnxDft<&[u8], B> {
        VecZnxDft {
            data: self.data.as_slice(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend> VecZnxDftToMut<B> for VecZnxDft<&mut [u8], B> {
    fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B> {
        VecZnxDft {
            data: self.data,
            n: self.n,
            cols: self.cols,
            size: self.size,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend> VecZnxDftToRef<B> for VecZnxDft<&mut [u8], B> {
    fn to_ref(&self) -> VecZnxDft<&[u8], B> {
        VecZnxDft {
            data: self.data,
            n: self.n,
            cols: self.cols,
            size: self.size,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend> VecZnxDftToRef<B> for VecZnxDft<&[u8], B> {
    fn to_ref(&self) -> VecZnxDft<&[u8], B> {
        VecZnxDft {
            data: self.data,
            n: self.n,
            cols: self.cols,
            size: self.size,
            _phantom: PhantomData,
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
