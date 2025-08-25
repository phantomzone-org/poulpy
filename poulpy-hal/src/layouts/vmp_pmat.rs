use std::marker::PhantomData;

use crate::{
    alloc_aligned,
    layouts::{Backend, Data, DataMut, DataRef, DataView, DataViewMut, ZnxInfos, ZnxView},
    oep::VmpPMatAllocBytesImpl,
};

#[derive(PartialEq, Eq)]
pub struct VmpPMat<D: Data, B: Backend> {
    data: D,
    n: usize,
    size: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    _phantom: PhantomData<B>,
}

impl<D: DataRef, B: Backend> ZnxView for VmpPMat<D, B> {
    type Scalar = B::ScalarPrep;
}

impl<D: Data, B: Backend> ZnxInfos for VmpPMat<D, B> {
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

impl<D: Data, B: Backend> DataView for VmpPMat<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, B: Backend> DataViewMut for VmpPMat<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: Data, B: Backend> VmpPMat<D, B> {
    pub fn cols_in(&self) -> usize {
        self.cols_in
    }

    pub fn cols_out(&self) -> usize {
        self.cols_out
    }
}

impl<D: DataRef + From<Vec<u8>>, B: Backend> VmpPMat<D, B>
where
    B: VmpPMatAllocBytesImpl<B>,
{
    pub fn alloc(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned(B::vmp_pmat_alloc_bytes_impl(
            n, rows, cols_in, cols_out, size,
        ));
        Self {
            data: data.into(),
            n,
            size,
            rows,
            cols_in,
            cols_out,
            _phantom: PhantomData,
        }
    }

    pub fn from_bytes(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::vmp_pmat_alloc_bytes_impl(n, rows, cols_in, cols_out, size));
        Self {
            data: data.into(),
            n,
            size,
            rows,
            cols_in,
            cols_out,
            _phantom: PhantomData,
        }
    }
}

pub type VmpPMatOwned<B> = VmpPMat<Vec<u8>, B>;
pub type VmpPMatRef<'a, B> = VmpPMat<&'a [u8], B>;

pub trait VmpPMatToRef<B: Backend> {
    fn to_ref(&self) -> VmpPMat<&[u8], B>;
}

impl<D: DataRef, B: Backend> VmpPMatToRef<B> for VmpPMat<D, B> {
    fn to_ref(&self) -> VmpPMat<&[u8], B> {
        VmpPMat {
            data: self.data.as_ref(),
            n: self.n,
            rows: self.rows,
            cols_in: self.cols_in,
            cols_out: self.cols_out,
            size: self.size,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub trait VmpPMatToMut<B: Backend> {
    fn to_mut(&mut self) -> VmpPMat<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> VmpPMatToMut<B> for VmpPMat<D, B> {
    fn to_mut(&mut self) -> VmpPMat<&mut [u8], B> {
        VmpPMat {
            data: self.data.as_mut(),
            n: self.n,
            rows: self.rows,
            cols_in: self.cols_in,
            cols_out: self.cols_out,
            size: self.size,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D: Data, B: Backend> VmpPMat<D, B> {
    pub fn from_data(data: D, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        Self {
            data,
            n,
            rows,
            cols_in,
            cols_out,
            size,
            _phantom: PhantomData,
        }
    }
}
