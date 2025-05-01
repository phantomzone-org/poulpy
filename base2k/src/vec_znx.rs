use crate::Backend;
use crate::Module;
use crate::assert_alignement;
use crate::cast_mut;
use crate::ffi::znx;
use crate::znx_base::{GetZnxBase, ZnxAlloc, ZnxBase, ZnxInfos, ZnxLayout, ZnxRsh, ZnxSliceSize, ZnxZero, switch_degree};
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
pub struct VecZnx {
    pub inner: ZnxBase,
}

impl GetZnxBase for VecZnx {
    fn znx(&self) -> &ZnxBase {
        &self.inner
    }

    fn znx_mut(&mut self) -> &mut ZnxBase {
        &mut self.inner
    }
}

impl ZnxInfos for VecZnx {}

impl ZnxSliceSize for VecZnx {
    fn sl(&self) -> usize {
        self.cols() * self.n()
    }
}

impl ZnxLayout for VecZnx {
    type Scalar = i64;
}

impl ZnxZero for VecZnx {}

impl ZnxRsh for VecZnx {}

impl<B: Backend> ZnxAlloc<B> for VecZnx {
    type Scalar = i64;

    fn from_bytes_borrow(module: &Module<B>, _rows: usize, cols: usize, size: usize, bytes: &mut [u8]) -> VecZnx {
        debug_assert_eq!(bytes.len(), Self::bytes_of(module, _rows, cols, size));
        VecZnx {
            inner: ZnxBase::from_bytes_borrow(module.n(), VEC_ZNX_ROWS, cols, size, bytes),
        }
    }

    fn bytes_of(module: &Module<B>, _rows: usize, cols: usize, size: usize) -> usize {
        debug_assert_eq!(
            _rows, VEC_ZNX_ROWS,
            "rows != {} not supported for VecZnx",
            VEC_ZNX_ROWS
        );
        module.n() * cols * size * size_of::<Self::Scalar>()
    }
}

/// Copies the coefficients of `a` on the receiver.
/// Copy is done with the minimum size matching both backing arrays.
/// Panics if the cols do not match.
pub fn copy_vec_znx_from(b: &mut VecZnx, a: &VecZnx) {
    assert_eq!(b.cols(), a.cols());
    let data_a: &[i64] = a.raw();
    let data_b: &mut [i64] = b.raw_mut();
    let size = min(data_b.len(), data_a.len());
    data_b[..size].copy_from_slice(&data_a[..size])
}

impl VecZnx {
    /// Truncates the precision of the [VecZnx] by k bits.
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: the base two logarithm of the coefficients decomposition.
    /// * `k`: the number of bits of precision to drop.
    pub fn trunc_pow2(&mut self, log_base2k: usize, k: usize, col: usize) {
        if k == 0 {
            return;
        }

        if !self.borrowing() {
            self.inner
                .data
                .truncate(self.n() * self.cols() * (self.size() - k / log_base2k));
        }

        self.inner.size -= k / log_base2k;

        let k_rem: usize = k % log_base2k;

        if k_rem != 0 {
            let mask: i64 = ((1 << (log_base2k - k_rem - 1)) - 1) << k_rem;
            self.at_mut(col, self.size() - 1)
                .iter_mut()
                .for_each(|x: &mut i64| *x &= mask)
        }
    }

    pub fn copy_from(&mut self, a: &Self) {
        copy_vec_znx_from(self, a);
    }

    pub fn normalize(&mut self, log_base2k: usize, col: usize, carry: &mut [u8]) {
        normalize(log_base2k, self, col, carry)
    }

    pub fn switch_degree(&self, col: usize, a: &mut Self, col_a: usize) {
        switch_degree(a, col_a, self, col)
    }

    // Prints the first `n` coefficients of each limb
    pub fn print(&self, n: usize, col: usize) {
        (0..self.size()).for_each(|j| println!("{}: {:?}", j, &self.at(col, j)[..n]));
    }
}

fn normalize_tmp_bytes(n: usize) -> usize {
    n * std::mem::size_of::<i64>()
}

fn normalize(log_base2k: usize, a: &mut VecZnx, a_col: usize, tmp_bytes: &mut [u8]) {
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
                log_base2k as u64,
                a.at_mut_ptr(a_col, i),
                carry_i64.as_mut_ptr(),
                a.at_mut_ptr(a_col, i),
                carry_i64.as_mut_ptr(),
            )
        });
    }
}
