use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::{
    alloc_aligned,
    layouts::{Backend, Data, DataMut, DataRef, DataView, DataViewMut, DigestU64, ZnxInfos, ZnxView},
};

/// Prepared (DFT-domain) scalar polynomial for scalar-vector products.
///
/// An `SvpPPol` holds a single polynomial in the backend's prepared
/// representation ([`Backend::ScalarPrep`]). It is used as the left
/// operand in [`SvpApplyDft`](crate::api::SvpApplyDft) to efficiently
/// multiply a scalar polynomial by each column of a [`VecZnxDft`](crate::layouts::VecZnxDft).
///
/// Create via [`SvpPrepare`](crate::api::SvpPrepare) from a
/// coefficient-domain [`ScalarZnx`](crate::layouts::ScalarZnx).
///
/// Ring degree `n` is always a power of two, so the DFT-domain layout has a
/// coefficient count that matches vector lane widths relative to buffer alignment.
#[repr(C)]
#[derive(PartialEq, Eq, Hash)]
pub struct SvpPPol<D: Data, B: Backend> {
    pub data: D,
    pub n: usize,
    pub cols: usize,
    pub _phantom: PhantomData<B>,
}

impl<D: DataRef, B: Backend> DigestU64 for SvpPPol<D, B> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n);
        h.write_usize(self.cols);
        h.finish()
    }
}

impl<D: DataRef, B: Backend> ZnxView for SvpPPol<D, B> {
    type Scalar = B::ScalarPrep;
}

impl<D: Data, B: Backend> ZnxInfos for SvpPPol<D, B> {
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

impl<D: Data, B: Backend> DataView for SvpPPol<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, B: Backend> DataViewMut for SvpPPol<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: Data + From<Vec<u8>>, B: Backend> SvpPPol<D, B> {
    pub fn alloc(n: usize, cols: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(B::bytes_of_svp_ppol(n, cols));
        Self {
            data: data.into(),
            n,
            cols,
            _phantom: PhantomData,
        }
    }
}

/// Owned `SvpPPol` backed by a `Vec<u8>`.
pub type SvpPPolOwned<B> = SvpPPol<Vec<u8>, B>;

/// Borrow an `SvpPPol` as a shared reference view.
pub trait SvpPPolToRef<B: Backend> {
    fn to_ref(&self) -> SvpPPol<&[u8], B>;
}

impl<D: DataRef, B: Backend> SvpPPolToRef<B> for SvpPPol<D, B> {
    fn to_ref(&self) -> SvpPPol<&[u8], B> {
        SvpPPol {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            _phantom: PhantomData,
        }
    }
}

/// Borrow an `SvpPPol` as a mutable reference view.
pub trait SvpPPolToMut<B: Backend> {
    fn to_mut(&mut self) -> SvpPPol<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> SvpPPolToMut<B> for SvpPPol<D, B> {
    fn to_mut(&mut self) -> SvpPPol<&mut [u8], B> {
        SvpPPol {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, B: Backend> SvpPPol<D, B> {
    pub fn from_data(data: D, n: usize, cols: usize) -> Self {
        Self {
            data,
            n,
            cols,
            _phantom: PhantomData,
        }
    }
}

impl<D: DataRef, B: Backend> fmt::Display for SvpPPol<D, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SvpPPol(n={}, cols={})", self.n, self.cols)?;

        for col in 0..self.cols {
            writeln!(f, "Column {col}:")?;
            let coeffs = self.at(col, 0);
            write!(f, "[")?;

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
        Ok(())
    }
}
