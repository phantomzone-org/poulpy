use crate::ffi::vec_znx_big;
use crate::znx_base::{ZnxInfos, ZnxView};
use crate::{Backend, DataView, DataViewMut, FFT64, Module, VecZnx, ZnxSliceSize, ZnxViewMut, ZnxZero, alloc_aligned};
use std::fmt;
use std::marker::PhantomData;

pub struct VecZnxBig<D, B: Backend> {
    data: D,
    n: usize,
    cols: usize,
    size: usize,
    _phantom: PhantomData<B>,
}

impl<D, B: Backend> ZnxInfos for VecZnxBig<D, B> {
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

impl<D> ZnxSliceSize for VecZnxBig<D, FFT64> {
    fn sl(&self) -> usize {
        self.n() * self.cols()
    }
}

impl<D, B: Backend> DataView for VecZnxBig<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D, B: Backend> DataViewMut for VecZnxBig<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: AsRef<[u8]>> ZnxView for VecZnxBig<D, FFT64> {
    type Scalar = i64;
}

pub(crate) fn bytes_of_vec_znx_big<B: Backend>(module: &Module<B>, cols: usize, size: usize) -> usize {
    unsafe { vec_znx_big::bytes_of_vec_znx_big(module.ptr, size as u64) as usize * cols }
}

impl<D: From<Vec<u8>>, B: Backend> VecZnxBig<D, B> {
    pub(crate) fn new(module: &Module<B>, cols: usize, size: usize) -> Self {
        let data = alloc_aligned::<u8>(bytes_of_vec_znx_big(module, cols, size));
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
        assert!(data.len() == bytes_of_vec_znx_big(module, cols, size));
        Self {
            data: data.into(),
            n: module.n(),
            cols,
            size,
            _phantom: PhantomData,
        }
    }
}

impl<D, B: Backend> VecZnxBig<D, B> {
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

impl<D: AsMut<[u8]> + AsRef<[u8]>> VecZnxBig<D, FFT64>
where
    VecZnxBig<D, FFT64>: VecZnxBigToMut<FFT64> + ZnxInfos,
{
    // Consumes the VecZnxBig to return a VecZnx.
    // Useful when no normalization is needed.
    pub fn to_vec_znx_small(self) -> VecZnx<D> {
        VecZnx {
            data: self.data,
            n: self.n,
            cols: self.cols,
            size: self.size,
        }
    }

    /// Extracts the a_col-th column of 'a' and stores it on the self_col-th column [Self].
    pub fn extract_column<C>(&mut self, self_col: usize, a: &C, a_col: usize)
    where
        C: VecZnxBigToRef<FFT64> + ZnxInfos,
    {
        #[cfg(debug_assertions)]
        {
            assert!(self_col < self.cols());
            assert!(a_col < a.cols());
        }

        let min_size: usize = self.size.min(a.size());
        let max_size: usize = self.size;

        let mut self_mut: VecZnxBig<&mut [u8], FFT64> = self.to_mut();
        let a_ref: VecZnxBig<&[u8], FFT64> = a.to_ref();

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

pub type VecZnxBigOwned<B> = VecZnxBig<Vec<u8>, B>;

pub trait VecZnxBigToRef<B: Backend> {
    fn to_ref(&self) -> VecZnxBig<&[u8], B>;
}

impl<D, B: Backend> VecZnxBigToRef<B> for VecZnxBig<D, B>
where
    D: AsRef<[u8]>,
    B: Backend,
{
    fn to_ref(&self) -> VecZnxBig<&[u8], B> {
        VecZnxBig {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub trait VecZnxBigToMut<B: Backend> {
    fn to_mut(&mut self) -> VecZnxBig<&mut [u8], B>;
}

impl<D, B: Backend> VecZnxBigToMut<B> for VecZnxBig<D, B>
where
    D: AsRef<[u8]> + AsMut<[u8]>,
    B: Backend,
{
    fn to_mut(&mut self) -> VecZnxBig<&mut [u8], B> {
        VecZnxBig {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D: AsRef<[u8]>> fmt::Display for VecZnxBig<D, FFT64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "VecZnxBig(n={}, cols={}, size={})",
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
