use std::marker::PhantomData;

use rand_distr::num_traits::Zero;

use crate::{
    alloc_aligned,
    hal::{
        api::{DataView, DataViewMut, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero},
        layouts::{Backend, Data, DataMut, DataRef},
    },
};

#[derive(PartialEq, Eq)]
pub struct VecZnxBig<D: Data, B: Backend> {
    pub(crate) data: D,
    pub(crate) n: usize,
    pub(crate) cols: usize,
    pub(crate) size: usize,
    pub(crate) max_size: usize,
    pub(crate) _phantom: PhantomData<B>,
}

impl<D: Data, B: Backend> ZnxInfos for VecZnxBig<D, B> {
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

impl<D: Data, B: Backend> DataView for VecZnxBig<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, B: Backend> DataViewMut for VecZnxBig<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

pub trait VecZnxBigBytesOf {
    fn bytes_of(n: usize, cols: usize, size: usize) -> usize;
}

impl<D: DataMut, B: Backend> ZnxZero for VecZnxBig<D, B>
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

impl<D: DataRef + From<Vec<u8>>, B: Backend> VecZnxBig<D, B>
where
    VecZnxBig<D, B>: VecZnxBigBytesOf,
{
    pub(crate) fn new(n: usize, cols: usize, size: usize) -> Self {
        let data = alloc_aligned::<u8>(Self::bytes_of(n, cols, size));
        Self {
            data: data.into(),
            n,
            cols,
            size,
            max_size: size,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn new_from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(n, cols, size));
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

impl<D: Data, B: Backend> VecZnxBig<D, B> {
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

pub type VecZnxBigOwned<B> = VecZnxBig<Vec<u8>, B>;

pub trait VecZnxBigToRef<B: Backend> {
    fn to_ref(&self) -> VecZnxBig<&[u8], B>;
}

impl<D: DataRef, B: Backend> VecZnxBigToRef<B> for VecZnxBig<D, B> {
    fn to_ref(&self) -> VecZnxBig<&[u8], B> {
        VecZnxBig {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            max_size: self.max_size,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub trait VecZnxBigToMut<B: Backend> {
    fn to_mut(&mut self) -> VecZnxBig<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> VecZnxBigToMut<B> for VecZnxBig<D, B> {
    fn to_mut(&mut self) -> VecZnxBig<&mut [u8], B> {
        VecZnxBig {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            max_size: self.max_size,
            _phantom: std::marker::PhantomData,
        }
    }
}
