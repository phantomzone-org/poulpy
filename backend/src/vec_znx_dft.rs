use std::marker::PhantomData;

use crate::ffi::vec_znx_dft;
use crate::znx_base::ZnxInfos;
use crate::{
    Backend, DataView, DataViewMut, FFT64, Module, VecZnxBig, ZnxSliceSize, ZnxView, ZnxViewMut, ZnxZero, alloc_aligned,
};
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

impl<D: AsMut<[u8]> + AsRef<[u8]>> VecZnxDft<D, FFT64>
where
    VecZnxDft<D, FFT64>: VecZnxDftToMut<FFT64>,
{
    /// Extracts the a_col-th column of 'a' and stores it on the self_col-th column [Self].
    pub fn extract_column<C: AsRef<[u8]>>(&mut self, self_col: usize, a: &VecZnxDft<C, FFT64>, a_col: usize)
    where
        VecZnxDft<C, FFT64>: VecZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(self_col < self.cols());
            assert!(a_col < a.cols());
        }

        let min_size: usize = self.size.min(a.size());
        let max_size: usize = self.size;

        let mut self_mut: VecZnxDft<&mut [u8], FFT64> = self.to_mut();
        let a_ref: VecZnxDft<&[u8], FFT64> = a.to_ref();

        (0..min_size).for_each(|i: usize| {
            self_mut
                .at_mut(self_col, i)
                .copy_from_slice(a_ref.at(a_col, i));
        });

        (min_size..max_size).for_each(|i| {
            self_mut.zero_at(self_col, i);
        });
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

impl<D, B: Backend> VecZnxDftToRef<B> for VecZnxDft<D, B>
where
    D: AsRef<[u8]>,
    B: Backend,
{
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

impl<D, B: Backend> VecZnxDftToMut<B> for VecZnxDft<D, B>
where
    D: AsRef<[u8]> + AsMut<[u8]>,
    B: Backend,
{
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
