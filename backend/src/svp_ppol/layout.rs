use crate::znx_base::ZnxInfos;
use crate::{Backend, DataView, DataViewMut, alloc_aligned};
use std::marker::PhantomData;

pub struct SvpPPol<D, B: Backend> {
    data: D,
    n: usize,
    cols: usize,
    _phantom: PhantomData<B>,
}

impl<D, B: Backend> ZnxInfos for SvpPPol<D, B> {
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

impl<D, B: Backend> DataView for SvpPPol<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D, B: Backend> DataViewMut for SvpPPol<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

pub trait SvpPPolBytesOf {
    fn bytes_of(n: usize, cols: usize) -> usize;
}

impl<D: From<Vec<u8>>, B: Backend> SvpPPol<D, B>
where
    SvpPPol<D, B>: SvpPPolBytesOf,
{
    pub(crate) fn alloc(n: usize, cols: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of(n, cols));
        Self {
            data: data.into(),
            n,
            cols,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn from_bytes(n: usize, cols: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(n, cols));
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

impl<D: AsRef<[u8]>, B: Backend> SvpPPolToRef<B> for SvpPPol<D, B> {
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

impl<D: AsMut<[u8]> + AsRef<[u8]>, B: Backend> SvpPPolToMut<B> for SvpPPol<D, B> {
    fn to_mut(&mut self) -> SvpPPol<&mut [u8], B> {
        SvpPPol {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            _phantom: PhantomData,
        }
    }
}

impl<D, B: Backend> SvpPPol<D, B> {
    pub(crate) fn from_data(data: D, n: usize, cols: usize) -> Self {
        Self {
            data,
            n,
            cols,
            _phantom: PhantomData,
        }
    }
}
