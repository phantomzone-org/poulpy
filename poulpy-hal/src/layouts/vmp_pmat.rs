use std::{
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::{
    alloc_aligned,
    layouts::{Backend, Data, DataMut, DataRef, DataView, DataViewMut, DigestU64, ZnxInfos, ZnxView},
    oep::VmpPMatAllocBytesImpl,
};

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

impl<D: DataRef, B: Backend> DigestU64 for VmpPMat<D, B> {
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

impl<D: DataRef + From<Vec<u8>>, B: Backend> VmpPMat<D, B>
where
    B: VmpPMatAllocBytesImpl<B>,
{
    pub fn alloc(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned(B::vmp_pmat_bytes_of_impl(n, rows, cols_in, cols_out, size));
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
        assert!(data.len() == B::vmp_pmat_bytes_of_impl(n, rows, cols_in, cols_out, size));
        crate::assert_alignment(data.as_ptr());
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

/// Owned `VmpPMat` backed by a `Vec<u8>`.
pub type VmpPMatOwned<B> = VmpPMat<Vec<u8>, B>;
/// Immutably borrowed `VmpPMat`.
pub type VmpPMatRef<'a, B> = VmpPMat<&'a [u8], B>;

/// Borrow a `VmpPMat` as a shared reference view.
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

/// Borrow a `VmpPMat` as a mutable reference view.
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
