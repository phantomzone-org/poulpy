use std::fmt;

use crate::{
    alloc_aligned,
    hal::api::{DataView, DataViewMut, ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut, ZnxZero},
};

/// [VecZnx] represents collection of contiguously stacked vector of small norm polynomials of
/// Zn\[X\] with [i64] coefficients.
/// A [VecZnx] is composed of multiple Zn\[X\] polynomials stored in a single contiguous array
/// in the memory.
///
/// # Example
///
/// Given 3 polynomials (a, b, c) of Zn\[X\], each with 4 columns, then the memory
/// layout is: `[a0, b0, c0, a1, b1, c1, a2, b2, c2, a3, b3, c3]`, where ai, bi, ci
/// are small polynomials of Zn\[X\].
#[derive(PartialEq, Eq)]
pub struct VecZnx<D> {
    pub(crate) data: D,
    pub(crate) n: usize,
    pub(crate) cols: usize,
    pub(crate) size: usize,
    pub(crate) max_size: usize,
}

impl<D> fmt::Debug for VecZnx<D>
where
    D: AsRef<[u8]>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D> ZnxInfos for VecZnx<D> {
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

impl<D> ZnxSliceSize for VecZnx<D> {
    fn sl(&self) -> usize {
        self.n() * self.cols()
    }
}

impl<D> DataView for VecZnx<D> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D> DataViewMut for VecZnx<D> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: AsRef<[u8]>> ZnxView for VecZnx<D> {
    type Scalar = i64;
}

impl VecZnx<Vec<u8>> {
    pub fn rsh_scratch_space(n: usize) -> usize {
        n * std::mem::size_of::<i64>()
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>> ZnxZero for VecZnx<D> {
    fn zero(&mut self) {
        self.raw_mut().fill(0)
    }
    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).fill(0);
    }
}

impl<D: AsRef<[u8]>> VecZnx<D> {
    pub fn bytes_of<Scalar: Sized>(n: usize, cols: usize, size: usize) -> usize {
        n * cols * size * size_of::<Scalar>()
    }
}

impl<D: From<Vec<u8>> + AsRef<[u8]>> VecZnx<D> {
    pub fn new<Scalar: Sized>(n: usize, cols: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of::<Scalar>(n, cols, size));
        Self {
            data: data.into(),
            n,
            cols,
            size,
            max_size: size,
        }
    }

    pub(crate) fn new_from_bytes<Scalar: Sized>(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of::<Scalar>(n, cols, size));
        Self {
            data: data.into(),
            n,
            cols,
            size,
            max_size: size,
        }
    }
}

impl<D> VecZnx<D> {
    pub(crate) fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            n,
            cols,
            size,
            max_size: size,
        }
    }
}

#[allow(dead_code)]
fn normalize_tmp_bytes(n: usize) -> usize {
    n * std::mem::size_of::<i64>()
}

impl<D: AsRef<[u8]>> fmt::Display for VecZnx<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "VecZnx(n={}, cols={}, size={})",
            self.n, self.cols, self.size
        )?;

        for col in 0..self.cols {
            writeln!(f, "Column {}:", col)?;
            for size in 0..self.size {
                let coeffs = self.at(col, size);
                write!(f, "  Size {}: [", size)?;

                let max_show = 100;
                let show_count = coeffs.len().min(max_show);

                for (i, &coeff) in coeffs.iter().take(show_count).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", coeff)?;
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

pub type VecZnxOwned = VecZnx<Vec<u8>>;
pub type VecZnxMut<'a> = VecZnx<&'a mut [u8]>;
pub type VecZnxRef<'a> = VecZnx<&'a [u8]>;

pub trait VecZnxToRef {
    fn to_ref(&self) -> VecZnx<&[u8]>;
}

impl<D> VecZnxToRef for VecZnx<D>
where
    D: AsRef<[u8]>,
{
    fn to_ref(&self) -> VecZnx<&[u8]> {
        VecZnx {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            max_size: self.max_size,
        }
    }
}

pub trait VecZnxToMut {
    fn to_mut(&mut self) -> VecZnx<&mut [u8]>;
}

impl<D> VecZnxToMut for VecZnx<D>
where
    D: AsRef<[u8]> + AsMut<[u8]>,
{
    fn to_mut(&mut self) -> VecZnx<&mut [u8]> {
        VecZnx {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            max_size: self.max_size,
        }
    }
}

impl<DataSelf: AsRef<[u8]>> VecZnx<DataSelf> {
    pub fn clone(&self) -> VecZnx<Vec<u8>> {
        let self_ref: VecZnx<&[u8]> = self.to_ref();
        VecZnx {
            data: self_ref.data.to_vec(),
            n: self_ref.n,
            cols: self_ref.cols,
            size: self_ref.size,
            max_size: self_ref.max_size,
        }
    }
}
