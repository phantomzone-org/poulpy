use std::{
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::layouts::{Backend, Data, DataView, DataViewMut, DigestU64, HostDataMut, HostDataRef, ZnxInfos, ZnxView};

/// Prepared (DFT-domain) polynomial matrix for vector-matrix products.
///
/// A `VmpPMat` stores a matrix of `rows * cols_in` entries, where each
/// entry is a [`VecZnxDft`](crate::layouts::VecZnxDft) of `cols_out`
/// columns and `size` limbs, all in the backend's prepared representation.
///
/// Used as the right operand in
/// [`VmpApplyDftToDft`](crate::api::VmpApplyDftToDft). Create via
/// [`VmpPrepare`](crate::api::VmpPrepare) from a coefficient-domain
/// [`MatZnx`](crate::layouts::MatZnx).
///
/// Ring degree `n` is always a power of two, so each prepared polynomial's DFT
/// coefficient count matches vector lane widths relative to buffer alignment.
#[repr(C)]
#[derive(PartialEq, Eq, Hash)]
pub struct VmpPMat<D: Data, B: Backend> {
    data: D,
    n: usize,
    size: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    _phantom: PhantomData<B>,
}

impl<D: HostDataRef, B: Backend> DigestU64 for VmpPMat<D, B> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n);
        h.write_usize(self.size);
        h.write_usize(self.rows);
        h.write_usize(self.cols_in);
        h.write_usize(self.cols_out);
        h.finish()
    }
}

impl<D: HostDataRef, B: Backend> ZnxView for VmpPMat<D, B> {
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

    fn poly_count(&self) -> usize {
        self.rows() * self.cols_in() * self.size() * self.cols_out()
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
    /// Returns the number of input columns.
    pub fn cols_in(&self) -> usize {
        self.cols_in
    }

    /// Returns the number of output columns.
    pub fn cols_out(&self) -> usize {
        self.cols_out
    }
}

impl<B: Backend> VmpPMat<<B as Backend>::OwnedBuf, B> {
    pub fn alloc(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        let data: <B as Backend>::OwnedBuf = B::alloc_bytes(B::bytes_of_vmp_pmat(n, rows, cols_in, cols_out, size));
        Self {
            data,
            n,
            size,
            rows,
            cols_in,
            cols_out,
            _phantom: PhantomData,
        }
    }
}

/// Owned `VmpPMat` backed by a backend-owned buffer.
pub type VmpPMatOwned<B> = VmpPMat<<B as Backend>::OwnedBuf, B>;
/// Immutably borrowed `VmpPMat`.
pub type VmpPMatRef<'a, B> = VmpPMat<&'a [u8], B>;
/// Shared backend-native borrow of a `VmpPMat`.
pub type VmpPMatBackendRef<'a, B> = VmpPMat<<B as Backend>::BufRef<'a>, B>;
/// Mutable backend-native borrow of a `VmpPMat`.
pub type VmpPMatBackendMut<'a, B> = VmpPMat<<B as Backend>::BufMut<'a>, B>;

/// Reborrow a mutable backend-native `VmpPMat` view as a shared backend-native view.
pub fn vmp_pmat_backend_ref_from_mut<'a, B: Backend>(pmat: &'a VmpPMatBackendMut<'a, B>) -> VmpPMatBackendRef<'a, B> {
    VmpPMat {
        data: B::view_ref_mut(&pmat.data),
        n: pmat.n,
        rows: pmat.rows,
        cols_in: pmat.cols_in,
        cols_out: pmat.cols_out,
        size: pmat.size,
        _phantom: PhantomData,
    }
}

pub fn vmp_pmat_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(
    pmat: &'a mut VmpPMatBackendMut<'b, B>,
) -> VmpPMatBackendMut<'a, B> {
    VmpPMat {
        data: B::view_mut_ref(&mut pmat.data),
        n: pmat.n,
        rows: pmat.rows,
        cols_in: pmat.cols_in,
        cols_out: pmat.cols_out,
        size: pmat.size,
        _phantom: PhantomData,
    }
}

/// Borrow a backend-owned `VmpPMat` using the backend's native view type.
pub trait VmpPMatToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> VmpPMatBackendRef<'_, B>;
}

impl<B: Backend> VmpPMatToBackendRef<B> for VmpPMat<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> VmpPMatBackendRef<'_, B> {
        VmpPMat {
            data: B::view(&self.data),
            n: self.n,
            rows: self.rows,
            cols_in: self.cols_in,
            cols_out: self.cols_out,
            size: self.size,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Mutably borrow a backend-owned `VmpPMat` using the backend's native view type.
pub trait VmpPMatToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B>;
}

impl<B: Backend> VmpPMatToBackendMut<B> for VmpPMat<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B> {
        VmpPMat {
            data: B::view_mut(&mut self.data),
            n: self.n,
            rows: self.rows,
            cols_in: self.cols_in,
            cols_out: self.cols_out,
            size: self.size,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Borrow a `VmpPMat` as a shared reference view.
pub trait VmpPMatToRef<B: Backend> {
    fn to_ref(&self) -> VmpPMat<&[u8], B>;
}

impl<D: HostDataRef, B: Backend> VmpPMatToRef<B> for VmpPMat<D, B> {
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

/// Borrow a `VmpPMat` as a mutable reference view.
pub trait VmpPMatToMut<B: Backend> {
    fn to_mut(&mut self) -> VmpPMat<&mut [u8], B>;
}

impl<D: HostDataMut, B: Backend> VmpPMatToMut<B> for VmpPMat<D, B> {
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
