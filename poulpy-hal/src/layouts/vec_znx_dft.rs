use std::{fmt, marker::PhantomData};

use rand_distr::num_traits::Zero;

use crate::{
    alloc_aligned,
    layouts::{
        Backend, Data, DataMut, DataRef, DataView, DataViewMut, VecZnxBig, ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut, ZnxZero,
    },
    oep::VecZnxDftAllocBytesImpl,
};
#[derive(PartialEq, Eq)]
pub struct VecZnxDft<D: Data, B: Backend> {
    pub data: D,
    pub n: usize,
    pub cols: usize,
    pub size: usize,
    pub max_size: usize,
    pub _phantom: PhantomData<B>,
}

impl<D: Data, B: Backend> ZnxSliceSize for VecZnxDft<D, B> {
    fn sl(&self) -> usize {
        B::layout_prep_word_count() * self.n() * self.cols()
    }
}

impl<D: DataRef, B: Backend> ZnxView for VecZnxDft<D, B> {
    type Scalar = B::ScalarPrep;
}

impl<D: Data, B: Backend> VecZnxDft<D, B> {
    pub fn into_big(self) -> VecZnxBig<D, B> {
        VecZnxBig::<D, B>::from_data(self.data, self.n, self.cols, self.size)
    }
}

impl<D: Data, B: Backend> ZnxInfos for VecZnxDft<D, B> {
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

impl<D: Data, B: Backend> DataView for VecZnxDft<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, B: Backend> DataViewMut for VecZnxDft<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: DataRef, B: Backend> VecZnxDft<D, B> {
    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

impl<D: DataMut, B: Backend> VecZnxDft<D, B> {
    pub fn set_size(&mut self, size: usize) {
        assert!(size <= self.max_size);
        self.size = size
    }
}

impl<D: DataMut, B: Backend> ZnxZero for VecZnxDft<D, B>
where
    Self: ZnxViewMut,
    <Self as ZnxView>::Scalar: Zero + Copy,
{
    fn zero(&mut self) {
        self.raw_mut().fill(<Self as ZnxView>::Scalar::zero())
    }
    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).fill(<Self as ZnxView>::Scalar::zero());
    }
}

impl<D: DataRef + From<Vec<u8>>, B: Backend> VecZnxDft<D, B>
where
    B: VecZnxDftAllocBytesImpl<B>,
{
    pub fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(B::vec_znx_dft_alloc_bytes_impl(n, cols, size));
        Self {
            data: data.into(),
            n,
            cols,
            size,
            max_size: size,
            _phantom: PhantomData,
        }
    }

    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::vec_znx_dft_alloc_bytes_impl(n, cols, size));
        Self {
            data: data.into(),
            n,
            cols,
            size,
            max_size: size,
            _phantom: PhantomData,
        }
    }
}

pub type VecZnxDftOwned<B> = VecZnxDft<Vec<u8>, B>;

impl<D: Data, B: Backend> VecZnxDft<D, B> {
    pub fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            n,
            cols,
            size,
            max_size: size,
            _phantom: PhantomData,
        }
    }
}

pub trait VecZnxDftToRef<B: Backend> {
    fn to_ref(&self) -> VecZnxDft<&[u8], B>;
}

impl<D: DataRef, B: Backend> VecZnxDftToRef<B> for VecZnxDft<D, B> {
    fn to_ref(&self) -> VecZnxDft<&[u8], B> {
        VecZnxDft {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            max_size: self.max_size,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub trait VecZnxDftToMut<B: Backend> {
    fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> VecZnxDftToMut<B> for VecZnxDft<D, B> {
    fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B> {
        VecZnxDft {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            max_size: self.max_size,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D: DataRef, B: Backend> fmt::Display for VecZnxDft<D, B> {
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
