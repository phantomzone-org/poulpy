use crate::Backend;
use crate::DataView;
use crate::DataViewMut;
use crate::Module;
use crate::ZnxView;
use crate::alloc_aligned;
use crate::assert_alignement;
use crate::cast_mut;
use crate::ffi::znx;
use crate::znx_base::{GetZnxBase, ZnxAlloc, ZnxBase, ZnxInfos, ZnxRsh, ZnxZero, switch_degree};
use std::{cmp::min, fmt};

// pub const VEC_ZNX_ROWS: usize = 1;

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
    n: usize,
    cols: usize,
    size: usize,
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
        self.cols() * self.n()
    }
}

impl<D> DataView for VecZnx<D> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D> DataViewMut for VecZnx<D> {
    fn data_mut(&self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: AsRef<[u8]>> ZnxView for VecZnx<D> {
    type Scalar = i64;
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> VecZnx<D> {
    pub fn normalize(&mut self, log_base2k: usize, col: usize, carry: &mut [u8]) {
        normalize(log_base2k, self, col, carry)
    }

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

        self.inner.size -= k / log_base2k;

        let k_rem: usize = k % log_base2k;

        if k_rem != 0 {
            let mask: i64 = ((1 << (log_base2k - k_rem - 1)) - 1) << k_rem;
            self.at_mut(col, self.size() - 1)
                .iter_mut()
                .for_each(|x: &mut i64| *x &= mask)
        }
    }

    /// Switches degree of from `a.n()` to `self.n()` into `self`
    pub fn switch_degree<Data: AsRef<[u8]>>(&mut self, col: usize, a: &Data, col_a: usize) {
        switch_degree(self, col_a, a, col)
    }

    // Prints the first `n` coefficients of each limb
    // pub fn print(&self, n: usize, col: usize) {
    //     (0..self.size()).for_each(|j| println!("{}: {:?}", j, &self.at(col, j)[..n]));
    // }
}

impl<D: From<Vec<u8>>> VecZnx<D> {
    pub(crate) fn bytes_of<Scalar: Sized>(n: usize, cols: usize, size: usize) -> usize {
        n * cols * size * size_of::<Scalar>()
    }

    pub(crate) fn new<Scalar: Sized>(n: usize, cols: usize, size: usize) -> Self {
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

//(Jay)TODO: Impl. truncate pow2 for Owned Vector

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

// if !self.borrowing() {
//     self.inner
//         .data
//         .truncate(self.n() * self.cols() * (self.size() - k / log_base2k));
// }

fn normalize_tmp_bytes(n: usize) -> usize {
    n * std::mem::size_of::<i64>()
}

fn normalize<D: AsMut<[u8]>>(log_base2k: usize, a: &mut VecZnx<D>, a_col: usize, tmp_bytes: &mut [u8]) {
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

// impl<B: Backend> ZnxAlloc<B> for VecZnx {
//     type Scalar = i64;

//     fn from_bytes_borrow(module: &Module<B>, _rows: usize, cols: usize, size: usize, bytes: &mut [u8]) -> VecZnx {
//         debug_assert_eq!(bytes.len(), Self::bytes_of(module, _rows, cols, size));
//         VecZnx {
//             inner: ZnxBase::from_bytes_borrow(module.n(), VEC_ZNX_ROWS, cols, size, bytes),
//         }
//     }

//     fn bytes_of(module: &Module<B>, _rows: usize, cols: usize, size: usize) -> usize {
//         debug_assert_eq!(
//             _rows, VEC_ZNX_ROWS,
//             "rows != {} not supported for VecZnx",
//             VEC_ZNX_ROWS
//         );
//         module.n() * cols * size * size_of::<Self::Scalar>()
//     }
// }

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
