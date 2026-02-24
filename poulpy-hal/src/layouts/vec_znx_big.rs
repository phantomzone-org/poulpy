use std::{
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use rand_distr::num_traits::Zero;
use std::fmt;

use crate::{
    alloc_aligned,
    layouts::{Backend, Data, DataMut, DataRef, DataView, DataViewMut, DigestU64, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero},
};

/// Extended-precision polynomial vector used as a result accumulator.
///
/// `VecZnxBig` has the same structural shape as [`VecZnx`](crate::layouts::VecZnx)
/// (`cols` columns, `size` limbs, ring degree `N`) but uses
/// [`Backend::ScalarBig`] as its coefficient type instead of `i64`.
/// The wider scalar type allows lossless accumulation of intermediate
/// products before normalization back to `i64` limbs.
///
/// The exact scalar width and memory layout are backend-specific.
#[repr(C)]
#[derive(PartialEq, Eq, Hash)]
pub struct VecZnxBig<D: Data, B: Backend> {
    pub data: D,
    pub n: usize,
    pub cols: usize,
    pub size: usize,
    pub max_size: usize,
    pub _phantom: PhantomData<B>,
}

impl<D: DataRef, B: Backend> DigestU64 for VecZnxBig<D, B> {
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

impl<D: DataRef, B: Backend> ZnxView for VecZnxBig<D, B> {
    type Scalar = B::ScalarBig;
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

impl<D: DataRef + From<Vec<u8>>, B: Backend> VecZnxBig<D, B> {
    pub fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data = alloc_aligned::<u8>(B::bytes_of_vec_znx_big(n, cols, size));
        Self {
            data: data.into(),
            n,
            cols,
            size,
            max_size: size,
            _phantom: PhantomData,
        }
    }

    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::bytes_of_vec_znx_big(n, cols, size));
        crate::assert_alignment(data.as_ptr());
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

/// Owned `VecZnxBig` backed by a `Vec<u8>`.
pub type VecZnxBigOwned<B> = VecZnxBig<Vec<u8>, B>;

/// Borrow a `VecZnxBig` as a shared reference view.
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

/// Borrow a `VecZnxBig` as a mutable reference view.
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

impl<D: DataRef, B: Backend> fmt::Display for VecZnxBig<D, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "VecZnxBig(n={}, cols={}, size={})", self.n, self.cols, self.size)?;

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
