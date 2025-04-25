#[derive(Copy, Clone)]
#[repr(C)]
pub struct LAYOUT{
    /// Ring degree.
    n: usize,
    /// Number of logical rows in the layout.
    rows: usize,
    /// Number of polynomials per row.
    cols: usize,
    /// Number of limbs per polynomial.
    size: usize,
    /// Whether limbs are interleaved across rows.
    ///
    /// For example, for (rows, cols, size) = (2, 2, 3):
    /// 
    /// - `true`: layout is ((a0, b0, a1, b1), (c0, d0, c1, d1))
    /// - `false`: layout is ((a0, a1, b0, b1), (c0, c1, d0, d1))
    interleaved : bool,
}

pub trait Infos {

    /// Returns the full layout.
    fn layout(&self) -> LAYOUT;

    /// Returns the ring degree of the polynomials.
    fn n(&self) -> usize;

    /// Returns the base two logarithm of the ring dimension of the polynomials.
    fn log_n(&self) -> usize;

    /// Returns the number of rows.
    fn rows(&self) -> usize;

    /// Returns the number of polynomials in each row.
    fn cols(&self) -> usize;

    /// Returns the number of limbs per polynomial.
    fn size(&self) -> usize;

    /// Whether limbs are interleaved across rows.
    fn interleaved(&self) -> bool;
}
