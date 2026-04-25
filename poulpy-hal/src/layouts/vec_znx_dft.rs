use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use rand_distr::num_traits::Zero;

use crate::layouts::{
    Backend, Data, DataView, DataViewMut, DigestU64, HostDataMut, HostDataRef, VecZnxBig, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero,
};

/// Polynomial vector in DFT (evaluation) domain.
///
/// `VecZnxDft` has the same structural shape as [`VecZnx`](crate::layouts::VecZnx)
/// but stores coefficients as [`Backend::ScalarPrep`] values in the
/// frequency domain rather than `i64` values in the coefficient domain.
///
/// Multiplication and scalar-vector/vector-matrix products are performed
/// in this representation to exploit FFT-based convolution. Use
/// [`VecZnxDftApply`](crate::api::VecZnxDftApply) /
/// [`VecZnxIdftApply`](crate::api::VecZnxIdftApply) to convert
/// between coefficient and DFT domains.
#[repr(C)]
#[derive(PartialEq, Eq)]
pub struct VecZnxDft<D: Data, B: Backend> {
    pub data: D,
    pub n: usize,
    pub cols: usize,
    pub size: usize,
    pub max_size: usize,
    pub _phantom: PhantomData<B>,
}

impl<D: HostDataRef, B: Backend> DigestU64 for VecZnxDft<D, B> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n);
        h.write_usize(self.cols);
        h.write_usize(self.size);
        h.write_usize(self.max_size);
        h.finish()
    }
}

impl<D: HostDataRef, B: Backend> ZnxView for VecZnxDft<D, B> {
    type Scalar = B::ScalarPrep;
}

impl<D: Data, B: Backend> VecZnxDft<D, B> {
    /// Reinterprets this DFT vector as a [`VecZnxBig`], consuming `self`.
    ///
    /// This is a zero-copy conversion that changes only the type tag;
    /// the underlying data buffer is moved as-is.
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

impl<D: HostDataRef, B: Backend> VecZnxDft<D, B> {
    /// Returns the allocated capacity (maximum number of limbs).
    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

impl<D: HostDataMut, B: Backend> VecZnxDft<D, B> {
    /// Sets the active limb count.
    ///
    /// # Panics
    ///
    /// Panics if `size > max_size`.
    pub fn set_size(&mut self, size: usize) {
        assert!(size <= self.max_size);
        self.size = size
    }
}

impl<D: HostDataMut, B: Backend> ZnxZero for VecZnxDft<D, B>
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

impl<B: Backend> VecZnxDft<<B as Backend>::OwnedBuf, B> {
    pub fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data: <B as Backend>::OwnedBuf = B::alloc_bytes(B::bytes_of_vec_znx_dft(n, cols, size));
        Self {
            data,
            n,
            cols,
            size,
            max_size: size,
            _phantom: PhantomData,
        }
    }

    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::bytes_of_vec_znx_dft(n, cols, size));
        let data: <B as Backend>::OwnedBuf = B::from_host_bytes(&data);
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

/// Owned `VecZnxDft` backed by a backend-owned buffer.
pub type VecZnxDftOwned<B> = VecZnxDft<<B as Backend>::OwnedBuf, B>;
/// Shared backend-native borrow of a `VecZnxDft`.
pub type VecZnxDftBackendRef<'a, B> = VecZnxDft<<B as Backend>::BufRef<'a>, B>;
/// Mutable backend-native borrow of a `VecZnxDft`.
pub type VecZnxDftBackendMut<'a, B> = VecZnxDft<<B as Backend>::BufMut<'a>, B>;

/// Reborrow a mutable backend-native `VecZnxDft` view as a shared backend-native view.
pub fn vec_znx_dft_backend_ref_from_mut<'a, 'b, B: Backend + 'b>(
    vec: &'a VecZnxDftBackendMut<'b, B>,
) -> VecZnxDftBackendRef<'a, B> {
    VecZnxDft {
        data: B::view_ref_mut(&vec.data),
        n: vec.n,
        cols: vec.cols,
        size: vec.size,
        max_size: vec.max_size,
        _phantom: PhantomData,
    }
}

pub fn vec_znx_dft_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(
    vec: &'a mut VecZnxDftBackendMut<'b, B>,
) -> VecZnxDftBackendMut<'a, B> {
    VecZnxDft {
        data: B::view_mut_ref(&mut vec.data),
        n: vec.n,
        cols: vec.cols,
        size: vec.size,
        max_size: vec.max_size,
        _phantom: PhantomData,
    }
}

impl<D: Data, B: Backend> VecZnxDft<D, B> {
    /// Constructs a `VecZnxDft` from raw parts without validation.
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

/// Borrow a backend-owned `VecZnxDft` using the backend's native view type.
pub trait VecZnxDftToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> VecZnxDftBackendRef<'_, B>;
}

impl<B: Backend> VecZnxDftToBackendRef<B> for VecZnxDft<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> VecZnxDftBackendRef<'_, B> {
        VecZnxDft {
            data: B::view(&self.data),
            n: self.n,
            cols: self.cols,
            size: self.size,
            max_size: self.max_size,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `VecZnxDft` as a shared backend-native view.
pub trait VecZnxDftReborrowBackendRef<B: Backend> {
    fn reborrow_backend_ref(&self) -> VecZnxDftBackendRef<'_, B>;
}

impl<'b, B: Backend + 'b> VecZnxDftReborrowBackendRef<B> for VecZnxDft<B::BufMut<'b>, B> {
    fn reborrow_backend_ref(&self) -> VecZnxDftBackendRef<'_, B> {
        vec_znx_dft_backend_ref_from_mut::<B>(self)
    }
}

/// Mutably borrow a backend-owned `VecZnxDft` using the backend's native view type.
pub trait VecZnxDftToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> VecZnxDftBackendMut<'_, B>;
}

impl<B: Backend> VecZnxDftToBackendMut<B> for VecZnxDft<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> VecZnxDftBackendMut<'_, B> {
        VecZnxDft {
            data: B::view_mut(&mut self.data),
            n: self.n,
            cols: self.cols,
            size: self.size,
            max_size: self.max_size,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `VecZnxDft` as a mutable backend-native view.
pub trait VecZnxDftReborrowBackendMut<B: Backend> {
    fn reborrow_backend_mut(&mut self) -> VecZnxDftBackendMut<'_, B>;
}

impl<'b, B: Backend + 'b> VecZnxDftReborrowBackendMut<B> for VecZnxDft<B::BufMut<'b>, B> {
    fn reborrow_backend_mut(&mut self) -> VecZnxDftBackendMut<'_, B> {
        vec_znx_dft_backend_mut_from_mut::<B>(self)
    }
}

/// Borrow a `VecZnxDft` as a shared reference view.
pub trait VecZnxDftToRef<B: Backend> {
    fn to_ref(&self) -> VecZnxDft<&[u8], B>;
}

impl<D: HostDataRef, B: Backend> VecZnxDftToRef<B> for VecZnxDft<D, B> {
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

/// Borrow a `VecZnxDft` as a mutable reference view.
pub trait VecZnxDftToMut<B: Backend> {
    fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B>;
}

impl<D: HostDataMut, B: Backend> VecZnxDftToMut<B> for VecZnxDft<D, B> {
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

impl<D: HostDataRef, B: Backend> fmt::Display for VecZnxDft<D, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "VecZnxDft(n={}, cols={}, size={})", self.n, self.cols, self.size)?;

        for col in 0..self.cols {
            writeln!(f, "Column {col}:")?;
            for size in 0..self.size {
                let coeffs = self.at(col, size);
                write!(f, "  Size {size}: [")?;

                let max_show = 100;
                let show_count = coeffs.len().min(max_show);

                for (i, &coeff) in coeffs.iter().take(show_count).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{coeff}")?;
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
