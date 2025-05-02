use crate::Backend;
use crate::alloc_aligned;

use crate::DataView;
use crate::DataViewMut;
use crate::Module;
use crate::assert_alignement;
use crate::cast_mut;
use crate::ffi::znx;
use crate::znx_base::{ZnxAlloc, ZnxBase, ZnxBasics, ZnxInfos, ZnxView, ZnxViewMut, switch_degree};
use std::cmp::min;

pub const VEC_ZNX_ROWS: usize = 1;

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
pub struct VecZnx<D> {
    data: D,
    cols: usize,
    size: usize,
    n: usize,
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

    fn sl(&self) -> usize {
        (self.cols() - 1) * self.n()
    }
}

impl<D: AsRef<[u8]>> ZnxView for VecZnx<D> {
    type Scalar = i64;
}

pub type VecZnxOwned = VecZnx<Vec<u8>>;
pub type VecZnxRef = VecZnx<[u8]>;
pub type VecZnxMut<'a> = VecZnx<&'a mut [u8]>;

impl VecZnxOwned {
    fn storage_bytes<B, S>(module: &Module<B>, cols: usize, size: usize) -> usize {
        module.n() * cols * size * size_of::<S>()
    }

    fn new<B, Scalar>(module: &Module<B>, cols: usize, size: usize) -> Self {
        Self {
            data: alloc_aligned::<u8>(Self::storage_bytes::<_, Scalar>(module, cols, size)),
            cols,
            size,
            n: module.n(),
        }
    }
}

impl<B> ZnxAlloc<B> for VecZnx {
    type Scalar = i64;

    // fn from_bytes_borrow(module: &Module<B>, _rows: usize, cols: usize, size: usize, bytes: &mut [u8]) -> VecZnx {
    //     debug_assert_eq!(bytes.len(), Self::bytes_of(module, _rows, cols, size));
    //     VecZnx {
    //         inner: ZnxBase::from_bytes_borrow(module.n(), VEC_ZNX_ROWS, cols, size, bytes),
    //     }
    // }
}
/// Truncates the precision of the [VecZnx] by k bits.
///
/// # Arguments
///
/// * `log_base2k`: the base two logarithm of the coefficients decomposition.
/// * `k`: the number of bits of precision to drop.
pub fn trunc_pow2(&mut self, log_base2k: usize, k: usize) {
    if k == 0 {
        return;
    }

    // if !self.borrowing() {
    self.data
        .as_mut()
        .truncate(self.n() * self.cols() * (self.size() - k / log_base2k));
    // }

    self.size -= k / log_base2k;

    let k_rem: usize = k % log_base2k;

    if k_rem != 0 {
        let mask: i64 = ((1 << (log_base2k - k_rem - 1)) - 1) << k_rem;
        self.at_limb_mut(self.size() - 1)
            .iter_mut()
            .for_each(|x: &mut i64| *x &= mask)
    }
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> VecZnx<D> {
    /// Copies the coefficients of `a` on the receiver.
    /// Copy is done with the minimum size matching both backing arrays.
    /// Panics if the cols do not match.
    fn copy_vec_znx_from(b: &mut Self, a: &Self) {
        assert_eq!(b.cols(), a.cols());
        let data_a: &[i64] = a.raw();
        let data_b: &mut [i64] = b.raw_mut();
        let size = min(data_b.len(), data_a.len());
        data_b[..size].copy_from_slice(&data_a[..size])
    }

    pub fn copy_from(&mut self, a: &Self) {
        Self::copy_vec_znx_from(self, a);
    }

    pub fn normalize(&mut self, log_base2k: usize, carry: &mut [u8]) {
        normalize(log_base2k, self, carry)
    }

    pub fn switch_degree(&self, col: usize, a: &mut Self, col_a: usize) {
        switch_degree(a, col_a, self, col)
    }
}

impl<D: AsRef<[u8]>> std::fmt::Display for VecZnx<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "VecZnx(n={}, cols={}, size={})",
            self.n(),
            self.cols(),
            self.size()
        )?;

        for i in 0..self.cols() {
            writeln!(f, "Polynomial {}:", i)?;
            for j in 0..self.size() {
                let poly = self.at(i, j);
                writeln!(f, "  Small Poly {}: {:?}", j, &poly[..50])?;
            }
        }

        Ok(())
    }
}

fn normalize_tmp_bytes(n: usize, size: usize) -> usize {
    n * size * std::mem::size_of::<i64>()
}

fn normalize<D: AsMut<[u8]> + AsRef<[u8]>>(log_base2k: usize, a: &mut VecZnx<D>, tmp_bytes: &mut [u8]) {
    let n: usize = a.n();
    let cols: usize = a.cols();

    debug_assert!(
        tmp_bytes.len() >= normalize_tmp_bytes(n, cols),
        "invalid tmp_bytes: tmp_bytes.len()={} < normalize_tmp_bytes({}, {})",
        tmp_bytes.len(),
        n,
        cols,
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
                (n * cols) as u64,
                log_base2k as u64,
                a.at_mut_ptr(0, i),
                carry_i64.as_mut_ptr(),
                a.at_mut_ptr(0, i),
                carry_i64.as_mut_ptr(),
            )
        });
    }
}
