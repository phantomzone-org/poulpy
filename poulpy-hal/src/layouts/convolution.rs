use std::marker::PhantomData;

use crate::{
    alloc_aligned,
    layouts::{Backend, Data, DataMut, DataRef, DataView, DataViewMut, ZnxInfos, ZnxView},
    oep::CnvPVecBytesOfImpl,
};

pub struct CnvPVecR<D: Data, BE: Backend> {
    data: D,
    n: usize,
    size: usize,
    _phantom: PhantomData<BE>,
}

impl<D: Data, BE: Backend> ZnxInfos for CnvPVecR<D, BE> {
    fn cols(&self) -> usize {
        1
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

impl<D: DataRef, BE: Backend> ZnxView for CnvPVecR<D, BE> {
    type Scalar = BE::ScalarPrep;
}

impl<D: DataRef + From<Vec<u8>>, B: Backend> CnvPVecR<D, B>
where
    B: CnvPVecBytesOfImpl,
{
    pub fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(B::bytes_of_cnv_pvec_right_impl(n, cols, size));
        Self {
            data: data.into(),
            n,
            size,
            _phantom: PhantomData,
        }
    }

    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::bytes_of_cnv_pvec_right_impl(n, cols, size));
        Self {
            data: data.into(),
            n,
            size,
            _phantom: PhantomData,
        }
    }
}

pub struct CnvPVecL<D: Data, BE: Backend> {
    data: D,
    n: usize,
    size: usize,
    _phantom: PhantomData<BE>,
}

impl<D: Data, BE: Backend> ZnxInfos for CnvPVecL<D, BE> {
    fn cols(&self) -> usize {
        1
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

impl<D: DataRef, BE: Backend> ZnxView for CnvPVecL<D, BE> {
    type Scalar = BE::ScalarPrep;
}

impl<D: DataRef + From<Vec<u8>>, B: Backend> CnvPVecL<D, B>
where
    B: CnvPVecBytesOfImpl,
{
    pub fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(B::bytes_of_cnv_pvec_left_impl(n, cols, size));
        Self {
            data: data.into(),
            n,
            size,
            _phantom: PhantomData,
        }
    }

    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::bytes_of_cnv_pvec_left_impl(n, cols, size));
        Self {
            data: data.into(),
            n,
            size,
            _phantom: PhantomData,
        }
    }
}

pub trait CnvPVecRToRef<BE: Backend> {
    fn to_ref(&self) -> CnvPVecR<&[u8], BE>;
}

impl<D: DataRef, BE: Backend> CnvPVecRToRef<BE> for CnvPVecR<D, BE> {
    fn to_ref(&self) -> CnvPVecR<&[u8], BE> {
        CnvPVecR {
            data: self.data.as_ref(),
            n: self.n,
            size: self.size,
            _phantom: self._phantom,
        }
    }
}

pub trait CnvPVecRToMut<BE: Backend> {
    fn to_mut(&mut self) -> CnvPVecR<&mut [u8], BE>;
}

impl<D: DataMut, BE: Backend> CnvPVecRToMut<BE> for CnvPVecR<D, BE> {
    fn to_mut(&mut self) -> CnvPVecR<&mut [u8], BE> {
        CnvPVecR {
            data: self.data.as_mut(),
            n: self.n,
            size: self.size,
            _phantom: self._phantom,
        }
    }
}

pub trait CnvPVecLToRef<BE: Backend> {
    fn to_ref(&self) -> CnvPVecL<&[u8], BE>;
}

impl<D: DataRef, BE: Backend> CnvPVecLToRef<BE> for CnvPVecL<D, BE> {
    fn to_ref(&self) -> CnvPVecL<&[u8], BE> {
        CnvPVecL {
            data: self.data.as_ref(),
            n: self.n,
            size: self.size,
            _phantom: self._phantom,
        }
    }
}

pub trait CnvPVecLToMut<BE: Backend> {
    fn to_mut(&mut self) -> CnvPVecL<&mut [u8], BE>;
}

impl<D: DataMut, BE: Backend> CnvPVecLToMut<BE> for CnvPVecL<D, BE> {
    fn to_mut(&mut self) -> CnvPVecL<&mut [u8], BE> {
        CnvPVecL {
            data: self.data.as_mut(),
            n: self.n,
            size: self.size,
            _phantom: self._phantom,
        }
    }
}
