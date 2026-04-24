use std::marker::PhantomData;

use crate::layouts::{Backend, Data, DataView, DataViewMut, HostDataMut, HostDataRef, ZnxInfos, ZnxView};

/// Prepared right operand for bivariate convolution.
///
/// Holds a polynomial vector in the backend's prepared representation,
/// ready to be used as the right operand of
/// [`Convolution::cnv_apply_dft`](crate::api::Convolution::cnv_apply_dft).
/// Created via [`Convolution::cnv_prepare_right`](crate::api::Convolution::cnv_prepare_right).
pub struct CnvPVecR<D: Data, BE: Backend> {
    data: D,
    n: usize,
    size: usize,
    cols: usize,
    _phantom: PhantomData<BE>,
}

impl<D: Data, BE: Backend> ZnxInfos for CnvPVecR<D, BE> {
    fn cols(&self) -> usize {
        self.cols
    }

    fn n(&self) -> usize {
        self.n
    }

    fn rows(&self) -> usize {
        1
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl<D: Data, BE: Backend> DataView for CnvPVecR<D, BE> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, B: Backend> DataViewMut for CnvPVecR<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef, BE: Backend> ZnxView for CnvPVecR<D, BE> {
    type Scalar = BE::ScalarPrep;
}

impl<B: Backend> CnvPVecR<B::OwnedBuf, B> {
    pub fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data: B::OwnedBuf = B::alloc_bytes(B::bytes_of_cnv_pvec_right(n, cols, size));
        Self {
            data,
            n,
            size,
            cols,
            _phantom: PhantomData,
        }
    }

    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::bytes_of_cnv_pvec_right(n, cols, size));
        let data: B::OwnedBuf = B::from_host_bytes(&data);
        Self {
            data,
            n,
            size,
            cols,
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, B: Backend> CnvPVecR<D, B> {
    pub fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            n,
            cols,
            size,
            _phantom: PhantomData,
        }
    }
}

/// Prepared left operand for bivariate convolution.
///
/// Holds a polynomial vector in the backend's prepared representation,
/// ready to be used as the left operand of
/// [`Convolution::cnv_apply_dft`](crate::api::Convolution::cnv_apply_dft).
/// Created via [`Convolution::cnv_prepare_left`](crate::api::Convolution::cnv_prepare_left).
pub struct CnvPVecL<D: Data, BE: Backend> {
    data: D,
    n: usize,
    size: usize,
    cols: usize,
    _phantom: PhantomData<BE>,
}

impl<D: Data, BE: Backend> ZnxInfos for CnvPVecL<D, BE> {
    fn cols(&self) -> usize {
        self.cols
    }

    fn n(&self) -> usize {
        self.n
    }

    fn rows(&self) -> usize {
        1
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl<D: Data, BE: Backend> DataView for CnvPVecL<D, BE> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, B: Backend> DataViewMut for CnvPVecL<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef, BE: Backend> ZnxView for CnvPVecL<D, BE> {
    type Scalar = BE::ScalarPrep;
}

impl<B: Backend> CnvPVecL<B::OwnedBuf, B> {
    pub fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data: B::OwnedBuf = B::alloc_bytes(B::bytes_of_cnv_pvec_left(n, cols, size));
        Self {
            data,
            n,
            size,
            cols,
            _phantom: PhantomData,
        }
    }

    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::bytes_of_cnv_pvec_left(n, cols, size));
        let data: B::OwnedBuf = B::from_host_bytes(&data);
        Self {
            data,
            n,
            size,
            cols,
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, B: Backend> CnvPVecL<D, B> {
    pub fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            n,
            cols,
            size,
            _phantom: PhantomData,
        }
    }
}

/// Borrow a `CnvPVecR` as a shared reference view.
pub type CnvPVecRBackendRef<'a, B> = CnvPVecR<<B as Backend>::BufRef<'a>, B>;
pub type CnvPVecRBackendMut<'a, B> = CnvPVecR<<B as Backend>::BufMut<'a>, B>;
pub type CnvPVecLBackendRef<'a, B> = CnvPVecL<<B as Backend>::BufRef<'a>, B>;
pub type CnvPVecLBackendMut<'a, B> = CnvPVecL<<B as Backend>::BufMut<'a>, B>;

/// Borrow a backend-owned `CnvPVecR` using the backend's native view type.
pub trait CnvPVecRToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> CnvPVecRBackendRef<'_, BE>;
}

impl<BE: Backend> CnvPVecRToBackendRef<BE> for CnvPVecR<BE::OwnedBuf, BE> {
    fn to_backend_ref(&self) -> CnvPVecRBackendRef<'_, BE> {
        CnvPVecR {
            data: BE::view(&self.data),
            n: self.n,
            size: self.size,
            cols: self.cols,
            _phantom: self._phantom,
        }
    }
}

/// Mutably borrow a backend-owned `CnvPVecR` using the backend's native view type.
pub trait CnvPVecRToBackendMut<BE: Backend> {
    fn to_backend_mut(&mut self) -> CnvPVecRBackendMut<'_, BE>;
}

impl<BE: Backend> CnvPVecRToBackendMut<BE> for CnvPVecR<BE::OwnedBuf, BE> {
    fn to_backend_mut(&mut self) -> CnvPVecRBackendMut<'_, BE> {
        CnvPVecR {
            data: BE::view_mut(&mut self.data),
            n: self.n,
            size: self.size,
            cols: self.cols,
            _phantom: self._phantom,
        }
    }
}

/// Borrow a `CnvPVecR` as a shared reference view.
pub trait CnvPVecRToRef<BE: Backend> {
    fn to_ref(&self) -> CnvPVecR<&[u8], BE>;
}

impl<D: HostDataRef, BE: Backend> CnvPVecRToRef<BE> for CnvPVecR<D, BE> {
    fn to_ref(&self) -> CnvPVecR<&[u8], BE> {
        CnvPVecR {
            data: self.data.as_ref(),
            n: self.n,
            size: self.size,
            cols: self.cols,
            _phantom: self._phantom,
        }
    }
}

/// Borrow a `CnvPVecR` as a mutable reference view.
pub trait CnvPVecRToMut<BE: Backend> {
    fn to_mut(&mut self) -> CnvPVecR<&mut [u8], BE>;
}

impl<D: HostDataMut, BE: Backend> CnvPVecRToMut<BE> for CnvPVecR<D, BE> {
    fn to_mut(&mut self) -> CnvPVecR<&mut [u8], BE> {
        CnvPVecR {
            data: self.data.as_mut(),
            n: self.n,
            size: self.size,
            cols: self.cols,
            _phantom: self._phantom,
        }
    }
}

/// Borrow a `CnvPVecL` as a shared reference view.
pub trait CnvPVecLToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> CnvPVecLBackendRef<'_, BE>;
}

impl<BE: Backend> CnvPVecLToBackendRef<BE> for CnvPVecL<BE::OwnedBuf, BE> {
    fn to_backend_ref(&self) -> CnvPVecLBackendRef<'_, BE> {
        CnvPVecL {
            data: BE::view(&self.data),
            n: self.n,
            size: self.size,
            cols: self.cols,
            _phantom: self._phantom,
        }
    }
}

/// Mutably borrow a backend-owned `CnvPVecL` using the backend's native view type.
pub trait CnvPVecLToBackendMut<BE: Backend> {
    fn to_backend_mut(&mut self) -> CnvPVecLBackendMut<'_, BE>;
}

impl<BE: Backend> CnvPVecLToBackendMut<BE> for CnvPVecL<BE::OwnedBuf, BE> {
    fn to_backend_mut(&mut self) -> CnvPVecLBackendMut<'_, BE> {
        CnvPVecL {
            data: BE::view_mut(&mut self.data),
            n: self.n,
            size: self.size,
            cols: self.cols,
            _phantom: self._phantom,
        }
    }
}

/// Borrow a `CnvPVecL` as a shared reference view.
pub trait CnvPVecLToRef<BE: Backend> {
    fn to_ref(&self) -> CnvPVecL<&[u8], BE>;
}

impl<D: HostDataRef, BE: Backend> CnvPVecLToRef<BE> for CnvPVecL<D, BE> {
    fn to_ref(&self) -> CnvPVecL<&[u8], BE> {
        CnvPVecL {
            data: self.data.as_ref(),
            n: self.n,
            size: self.size,
            cols: self.cols,
            _phantom: self._phantom,
        }
    }
}

/// Borrow a `CnvPVecL` as a mutable reference view.
pub trait CnvPVecLToMut<BE: Backend> {
    fn to_mut(&mut self) -> CnvPVecL<&mut [u8], BE>;
}

impl<D: HostDataMut, BE: Backend> CnvPVecLToMut<BE> for CnvPVecL<D, BE> {
    fn to_mut(&mut self) -> CnvPVecL<&mut [u8], BE> {
        CnvPVecL {
            data: self.data.as_mut(),
            n: self.n,
            size: self.size,
            cols: self.cols,
            _phantom: self._phantom,
        }
    }
}
