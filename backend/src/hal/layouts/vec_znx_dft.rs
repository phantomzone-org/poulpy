use std::marker::PhantomData;

use rand_distr::num_traits::Zero;

use crate::{
    alloc_aligned,
    hal::{
        api::{DataView, DataViewMut, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero},
        layouts::{Backend, Data, DataMut, DataRef, VecZnxBig},
    },
};
#[derive(PartialEq, Eq)]
pub struct VecZnxDft<D: Data, B: Backend> {
    pub(crate) data: D,
    pub(crate) n: usize,
    pub(crate) cols: usize,
    pub(crate) size: usize,
    pub(crate) max_size: usize,
    pub(crate) _phantom: PhantomData<B>,
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

pub trait VecZnxDftBytesOf {
    fn bytes_of(n: usize, cols: usize, size: usize) -> usize;
}

impl<D: DataRef + From<Vec<u8>>, B: Backend> VecZnxDft<D, B>
where
    VecZnxDft<D, B>: VecZnxDftBytesOf,
{
    pub(crate) fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of(n, cols, size));
        Self {
            data: data.into(),
            n: n,
            cols,
            size,
            max_size: size,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(n, cols, size));
        Self {
            data: data.into(),
            n: n,
            cols,
            size,
            max_size: size,
            _phantom: PhantomData,
        }
    }
}

pub type VecZnxDftOwned<B> = VecZnxDft<Vec<u8>, B>;

impl<D: Data, B: Backend> VecZnxDft<D, B> {
    pub(crate) fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
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
