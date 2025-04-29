use crate::Backend;
use crate::ZnxBase;
use crate::cast_mut;
use crate::ffi::znx;
use crate::switch_degree;
use crate::{Module, ZnxBasics, ZnxInfos, ZnxLayout};
use crate::{alloc_aligned, assert_alignement};
use std::cmp::min;

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
#[derive(Clone)]
pub struct VecZnx {
    /// Polynomial degree.
    pub n: usize,

    /// The number of polynomials
    pub cols: usize,

    /// The number of size per polynomial (a.k.a small polynomials).
    pub size: usize,

    /// Polynomial coefficients, as a contiguous array. Each col is equally spaced by n.
    pub data: Vec<i64>,

    /// Pointer to data (data can be enpty if [VecZnx] borrows space instead of owning it).
    pub ptr: *mut i64,
}

impl ZnxInfos for VecZnx {
    fn n(&self) -> usize {
        self.n
    }

    fn rows(&self) -> usize {
        1
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl ZnxLayout for VecZnx {
    type Scalar = i64;

    fn as_ptr(&self) -> *const Self::Scalar {
        self.ptr
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Scalar {
        self.ptr
    }
}

impl ZnxBasics for VecZnx {}

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

impl<B: Backend> ZnxBase<B> for VecZnx {
    type Scalar = i64;

    /// Allocates a new [VecZnx] composed of #size polynomials of Z\[X\].
    fn new(module: &Module<B>, cols: usize, size: usize) -> Self {
        let n: usize = module.n();
        #[cfg(debug_assertions)]
        {
            assert!(n > 0);
            assert!(n & (n - 1) == 0);
            assert!(cols > 0);
            assert!(size > 0);
        }
        let mut data: Vec<i64> = alloc_aligned::<i64>(Self::bytes_of(module, cols, size));
        let ptr: *mut i64 = data.as_mut_ptr();
        Self {
            n: n,
            cols: cols,
            size: size,
            data: data,
            ptr: ptr,
        }
    }

    fn bytes_of(module: &Module<B>, cols: usize, size: usize) -> usize {
        module.n() * cols * size * size_of::<i64>()
    }

    /// Returns a new struct implementing [VecZnx] with the provided data as backing array.
    ///
    /// The struct will take ownership of buf[..[Self::bytes_of]]
    ///
    /// User must ensure that data is properly alligned and that
    /// the size of data is equal to [Self::bytes_of].
    fn from_bytes(module: &Module<B>, cols: usize, size: usize, bytes: &mut [u8]) -> Self {
        let n: usize = module.n();
        #[cfg(debug_assertions)]
        {
            assert!(cols > 0);
            assert!(size > 0);
            assert_eq!(bytes.len(), Self::bytes_of(module, cols, size));
            assert_alignement(bytes.as_ptr());
        }
        unsafe {
            let bytes_i64: &mut [i64] = cast_mut::<u8, i64>(bytes);
            let ptr: *mut i64 = bytes_i64.as_mut_ptr();
            Self {
                n: n,
                cols: cols,
                size: size,
                data: Vec::from_raw_parts(ptr, bytes.len(), bytes.len()),
                ptr: ptr,
            }
        }
    }

    fn from_bytes_borrow(module: &Module<B>, cols: usize, size: usize, bytes: &mut [u8]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(cols > 0);
            assert!(size > 0);
            assert!(bytes.len() >= Self::bytes_of(module, cols, size));
            assert_alignement(bytes.as_ptr());
        }
        Self {
            n: module.n(),
            cols: cols,
            size: size,
            data: Vec::new(),
            ptr: bytes.as_mut_ptr() as *mut i64,
        }
    }
}

impl VecZnx {
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

        if !self.borrowing() {
            self.data
                .truncate(self.n() * self.cols() * (self.size() - k / log_base2k));
        }

        self.size -= k / log_base2k;

        let k_rem: usize = k % log_base2k;

        if k_rem != 0 {
            let mask: i64 = ((1 << (log_base2k - k_rem - 1)) - 1) << k_rem;
            self.at_limb_mut(self.size() - 1)
                .iter_mut()
                .for_each(|x: &mut i64| *x &= mask)
        }
    }

    pub fn copy_from(&mut self, a: &Self) {
        copy_vec_znx_from(self, a);
    }

    pub fn borrowing(&self) -> bool {
        self.data.len() == 0
    }

    pub fn normalize(&mut self, log_base2k: usize, carry: &mut [u8]) {
        normalize(log_base2k, self, carry)
    }

    pub fn switch_degree(&self, col: usize, a: &mut Self, col_a: usize) {
        switch_degree(a, col_a, self, col)
    }

    // Prints the first `n` coefficients of each limb
    pub fn print(&self, n: usize) {
        (0..self.size()).for_each(|i| println!("{}: {:?}", i, &self.at_limb(i)[..n]))
    }
}

fn normalize_tmp_bytes(n: usize, size: usize) -> usize {
    n * size * std::mem::size_of::<i64>()
}

fn normalize(log_base2k: usize, a: &mut VecZnx, tmp_bytes: &mut [u8]) {
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
