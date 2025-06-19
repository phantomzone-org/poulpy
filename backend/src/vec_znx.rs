use itertools::izip;

use crate::DataView;
use crate::DataViewMut;
use crate::ScalarZnx;
use crate::Scratch;
use crate::ZnxSliceSize;
use crate::ZnxZero;
use crate::alloc_aligned;
use crate::assert_alignement;
use crate::cast_mut;
use crate::ffi::znx;
use crate::znx_base::{ZnxInfos, ZnxView, ZnxViewMut};
use std::{cmp::min, fmt};

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
    pub data: D,
    pub n: usize,
    pub cols: usize,
    pub size: usize,
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

impl<D: AsMut<[u8]> + AsRef<[u8]>> VecZnx<D> {
    /// Truncates the precision of the [VecZnx] by k bits.
    ///
    /// # Arguments
    ///
    /// * `basek`: the base two logarithm of the coefficients decomposition.
    /// * `k`: the number of bits of precision to drop.
    pub fn trunc_pow2(&mut self, basek: usize, k: usize, col: usize) {
        if k == 0 {
            return;
        }

        self.size -= k / basek;

        let k_rem: usize = k % basek;

        if k_rem != 0 {
            let mask: i64 = ((1 << (basek - k_rem - 1)) - 1) << k_rem;
            self.at_mut(col, self.size() - 1)
                .iter_mut()
                .for_each(|x: &mut i64| *x &= mask)
        }
    }

    pub fn rsh(&mut self, basek: usize, k: usize, scratch: &mut Scratch) {
        let n: usize = self.n();
        let cols: usize = self.cols();
        let size: usize = self.size();
        let steps: usize = k / basek;

        self.raw_mut().rotate_right(n * steps * cols);
        (0..cols).for_each(|i| {
            (0..steps).for_each(|j| {
                self.zero_at(i, j);
            })
        });

        let k_rem: usize = k % basek;

        if k_rem != 0 {
            let (carry, _) = scratch.tmp_slice::<i64>(n);
            let shift = i64::BITS as usize - k_rem;
            (0..cols).for_each(|i| {
                carry.fill(0);
                (steps..size).for_each(|j| {
                    izip!(carry.iter_mut(), self.at_mut(i, j).iter_mut()).for_each(|(ci, xi)| {
                        *xi += *ci << basek;
                        *ci = (*xi << shift) >> shift;
                        *xi = (*xi - *ci) >> k_rem;
                    });
                });
            })
        }
    }

    pub fn lsh(&mut self, basek: usize, k: usize, scratch: &mut Scratch) {
        let n: usize = self.n();
        let cols: usize = self.cols();
        let size: usize = self.size();
        let steps: usize = k / basek;

        self.raw_mut().rotate_left(n * steps * cols);
        (0..cols).for_each(|i| {
            (size - steps..size).for_each(|j| {
                self.zero_at(i, j);
            })
        });

        let k_rem: usize = k % basek;

        if k_rem != 0 {
            let shift: usize = i64::BITS as usize - k_rem;
            let (tmp_bytes, _) = scratch.tmp_slice::<u8>(n * size_of::<i64>());
            (0..cols).for_each(|i| {
                (0..steps).for_each(|j| {
                    self.at_mut(i, j).iter_mut().for_each(|xi| {
                        *xi <<= shift;
                    });
                });
                normalize(basek, self, i, tmp_bytes);
            });
        }
    }
}

impl<D: From<Vec<u8>>> VecZnx<D> {
    pub(crate) fn bytes_of<Scalar: Sized>(n: usize, cols: usize, size: usize) -> usize {
        n * cols * size * size_of::<Scalar>()
    }

    pub fn new<Scalar: Sized>(n: usize, cols: usize, size: usize) -> Self {
        let data = alloc_aligned::<u8>(Self::bytes_of::<Scalar>(n, cols, size));
        Self {
            data: data.into(),
            n,
            cols,
            size,
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
        }
    }

    pub fn to_scalar_znx(self) -> ScalarZnx<D> {
        debug_assert_eq!(
            self.size, 1,
            "cannot convert VecZnx to ScalarZnx if cols: {} != 1",
            self.cols
        );
        ScalarZnx {
            data: self.data,
            n: self.n,
            cols: self.cols,
        }
    }
}

/// Copies the coefficients of `a` on the receiver.
/// Copy is done with the minimum size matching both backing arrays.
/// Panics if the cols do not match.
pub fn copy_vec_znx_from<DataMut, Data>(b: &mut VecZnx<DataMut>, a: &VecZnx<Data>)
where
    DataMut: AsMut<[u8]> + AsRef<[u8]>,
    Data: AsRef<[u8]>,
{
    assert_eq!(b.cols(), a.cols());
    let data_a: &[i64] = a.raw();
    let data_b: &mut [i64] = b.raw_mut();
    let size = min(data_b.len(), data_a.len());
    data_b[..size].copy_from_slice(&data_a[..size])
}

#[allow(dead_code)]
fn normalize_tmp_bytes(n: usize) -> usize {
    n * std::mem::size_of::<i64>()
}

impl<D: AsRef<[u8]> + AsMut<[u8]>> VecZnx<D> {
    pub fn normalize(&mut self, basek: usize, a_col: usize, tmp_bytes: &mut [u8]) {
        normalize(basek, self, a_col, tmp_bytes);
    }
}

fn normalize<D: AsMut<[u8]> + AsRef<[u8]>>(basek: usize, a: &mut VecZnx<D>, a_col: usize, tmp_bytes: &mut [u8]) {
    let n: usize = a.n();

    debug_assert!(
        tmp_bytes.len() >= normalize_tmp_bytes(n),
        "invalid tmp_bytes: tmp_bytes.len()={} < normalize_tmp_bytes({})",
        tmp_bytes.len(),
        n,
    );

    #[cfg(debug_assertions)]
    {
        assert_alignement(tmp_bytes.as_ptr())
    }

    let carry_i64: &mut [i64] = cast_mut(tmp_bytes);

    unsafe {
        znx::znx_zero_i64_ref(n as u64, carry_i64.as_mut_ptr());
        (0..a.size()).rev().for_each(|i| {
            znx::znx_normalize(
                n as u64,
                basek as u64,
                a.at_mut_ptr(a_col, i),
                carry_i64.as_mut_ptr(),
                a.at_mut_ptr(a_col, i),
                carry_i64.as_mut_ptr(),
            )
        });
    }
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> VecZnx<D>
where
    VecZnx<D>: VecZnxToMut + ZnxInfos,
{
    /// Extracts the a_col-th column of 'a' and stores it on the self_col-th column [Self].
    pub fn extract_column<R>(&mut self, self_col: usize, a: &VecZnx<R>, a_col: usize)
    where
        R: AsRef<[u8]>,
        VecZnx<R>: VecZnxToRef + ZnxInfos,
    {
        #[cfg(debug_assertions)]
        {
            assert!(self_col < self.cols());
            assert!(a_col < a.cols());
        }

        let min_size: usize = self.size.min(a.size());
        let max_size: usize = self.size;

        let mut self_mut: VecZnx<&mut [u8]> = self.to_mut();
        let a_ref: VecZnx<&[u8]> = a.to_ref();

        (0..min_size).for_each(|i: usize| {
            self_mut
                .at_mut(self_col, i)
                .copy_from_slice(a_ref.at(a_col, i));
        });

        (min_size..max_size).for_each(|i| {
            self_mut.zero_at(self_col, i);
        });
    }
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
        }
    }
}
