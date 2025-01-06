use crate::modulus::barrett::Barrett;
use crate::modulus::montgomery::Montgomery;
use crate::modulus::prime::Prime;
use crate::modulus::{REDUCEMOD, NONE};
use crate::modulus::{ScalarOperations, VectorOperations};
use crate::{apply_sv, apply_svv, apply_v, apply_vsv, apply_vv, apply_vvsv, apply_vvv, apply_ssv, apply_vssv};
use itertools::izip;

impl ScalarOperations<u64> for Prime<u64> {
    /// Applies a modular reduction on x based on REDUCE:
    /// - LAZY: no modular reduction.
    /// - ONCE: subtracts q if x >= q.
    /// - TWO: subtracts 2q if x >= 2q.
    /// - FOUR: subtracts 4q if x >= 4q.
    /// - BARRETT: maps x to x mod q using Barrett reduction.
    /// - BARRETTLAZY: maps x to x mod q using Barrett reduction with values in [0, 2q-1].
    #[inline(always)]
    fn sa_reduce_into_sa<const REDUCE: REDUCEMOD>(&self, a: &mut u64) {
        self.montgomery.reduce_assign::<REDUCE>(a);
    }

    #[inline(always)]
    fn sa_add_sb_into_sc<const REDUCE: REDUCEMOD>(&self, a: &u64, b: &u64, c: &mut u64) {
        *c = a.wrapping_add(*b);
        self.sa_reduce_into_sa::<REDUCE>(c);
    }

    #[inline(always)]
    fn sa_add_sb_into_sb<const REDUCE: REDUCEMOD>(&self, a: &u64, b: &mut u64) {
        *b = a.wrapping_add(*b);
        self.sa_reduce_into_sa::<REDUCE>(b);
    }

    #[inline(always)]
    fn sa_sub_sb_into_sc<const SBRANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &u64, b: &u64, c: &mut u64) {
        match SBRANGE{
            1 =>{*c = *a + self.q - *b}
            2 =>{*c = *a + self.two_q - *b}
            4 =>{*c = *a + self.four_q - *b}
            _ => unreachable!("invalid SBRANGE argument"),
        }
        self.sa_reduce_into_sa::<REDUCE>(c)
    }

    #[inline(always)]
    fn sa_sub_sb_into_sb<const SBRANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &u64, b: &mut u64) {
        match SBRANGE{
            1 =>{*b = *a + self.q - *b}
            2 =>{*b = *a + self.two_q - *b}
            4 =>{*b = *a + self.four_q - *b}
            _ => unreachable!("invalid SBRANGE argument"),
        }
        self.sa_reduce_into_sa::<REDUCE>(b)
    }

    #[inline(always)]
    fn sa_neg_into_sa<const SBRANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &mut u64) {
        match SBRANGE{
            1 =>{*a = self.q - *a}
            2 =>{*a = self.two_q - *a}
            4 =>{*a = self.four_q - *a}
            _ => unreachable!("invalid SBRANGE argument"),
        }
        self.sa_reduce_into_sa::<REDUCE>(a)
    }

    #[inline(always)]
    fn sa_neg_into_sb<const SBRANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &u64, b: &mut u64) {
        match SBRANGE{
            1 =>{*b = self.q - *a}
            2 =>{*b = self.two_q - *a}
            4 =>{*b = self.four_q - *a}
            _ => unreachable!("invalid SBRANGE argument"),
        }
        self.sa_reduce_into_sa::<REDUCE>(b)
    }

    #[inline(always)]
    fn sa_prep_mont_into_sb<const REDUCE: REDUCEMOD>(&self, a: &u64, b: &mut Montgomery<u64>) {
        self.montgomery.prepare_assign::<REDUCE>(*a, b);
    }

    #[inline(always)]
    fn sa_mont_mul_sb_into_sc<const REDUCE: REDUCEMOD>(
        &self,
        a: &Montgomery<u64>,
        b: &u64,
        c: &mut u64,
    ) {
        *c = self.montgomery.mul_external::<REDUCE>(*a, *b);
    }

    #[inline(always)]
    fn sa_mont_mul_sb_into_sb<const REDUCE: REDUCEMOD>(&self, a: &Montgomery<u64>, b: &mut u64) {
        self.montgomery.mul_external_assign::<REDUCE>(*a, b);
    }

    #[inline(always)]
    fn sa_barrett_mul_sb_into_sc<const REDUCE: REDUCEMOD>(
        &self,
        a: &Barrett<u64>,
        b: &u64,
        c: &mut u64,
    ) {
        *c = self.barrett.mul_external::<REDUCE>(*a, *b);
    }

    #[inline(always)]
    fn sa_barrett_mul_sb_into_sb<const REDUCE: REDUCEMOD>(&self, a: &Barrett<u64>, b: &mut u64) {
        self.barrett.mul_external_assign::<REDUCE>(*a, b);
    }

    #[inline(always)]
    fn sa_sub_sb_mul_sc_into_sd<const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        b: &u64,
        c: &Barrett<u64>,
        d: &mut u64,
    ) {
        match VBRANGE{
            1 =>{*d = a + self.q - b}
            2 =>{*d = a + self.two_q - b}
            4 =>{*d = a + self.four_q - b}
            _ => unreachable!("invalid SBRANGE argument"),
        }
        self.barrett.mul_external_assign::<REDUCE>(*c, d);
    }

    #[inline(always)]
    fn sa_sub_sb_mul_sc_into_sb<const SBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        c: &Barrett<u64>,
        b: &mut u64,
    ) {
        self.sa_sub_sb_into_sb::<SBRANGE, NONE>(a, b);
        self.barrett.mul_external_assign::<REDUCE>(*c, b);
    }

    #[inline(always)]
    fn sa_add_sb_mul_sc_into_sd<const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        b: &u64,
        c: &Barrett<u64>,
        d: &mut u64
    ) {
        *d = self.barrett.mul_external::<REDUCE>(*c, *a + *b);
    }

    #[inline(always)]
    fn sa_add_sb_mul_sc_into_sa<const REDUCE: REDUCEMOD>(
        &self,
        b: &u64,
        c: &Barrett<u64>,
        a: &mut u64
    ) {
        *a = self.barrett.mul_external::<REDUCE>(*c, *a + *b);
    }

}

impl VectorOperations<u64> for Prime<u64> {
    /// Applies a modular reduction on x based on REDUCE:
    /// - LAZY: no modular reduction.
    /// - ONCE: subtracts q if x >= q.
    /// - TWO: subtracts 2q if x >= 2q.
    /// - FOUR: subtracts 4q if x >= 4q.
    /// - BARRETT: maps x to x mod q using Barrett reduction.
    /// - BARRETTLAZY: maps x to x mod q using Barrett reduction with values in [0, 2q-1].
    #[inline(always)]
    fn va_reduce_into_va<const CHUNK: usize, const REDUCE: REDUCEMOD>(&self, a: &mut [u64]) {
        apply_v!(self, Self::sa_reduce_into_sa::<REDUCE>, a, CHUNK);
    }

    #[inline(always)]
    fn va_add_vb_into_vc<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &[u64],
        c: &mut [u64],
    ) {
        apply_vvv!(self, Self::sa_add_sb_into_sc::<REDUCE>, a, b, c, CHUNK);
    }

    #[inline(always)]
    fn va_add_vb_into_vb<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &mut [u64],
    ) {
        apply_vv!(self, Self::sa_add_sb_into_sb::<REDUCE>, a, b, CHUNK);
    }

    #[inline(always)]
    fn va_add_sb_into_va<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        b: &u64,
        a: &mut [u64],
    ) {
        apply_sv!(self, Self::sa_add_sb_into_sb::<REDUCE>, b, a, CHUNK);
    }

    #[inline(always)]
    fn va_add_sb_into_vc<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &u64,
        c: &mut [u64],
    ) {
        apply_vsv!(self, Self::sa_add_sb_into_sc::<REDUCE>, a, b, c, CHUNK);
    }

    #[inline(always)]
    fn va_sub_vb_into_vc<const CHUNK: usize, const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &[u64],
        c: &mut [u64],
    ) {
        apply_vvv!(self, Self::sa_sub_sb_into_sc::<VBRANGE, REDUCE>, a, b, c, CHUNK);
    }

    #[inline(always)]
    fn va_sub_vb_into_vb<const CHUNK: usize, const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &mut [u64],
    ) {
        apply_vv!(self, Self::sa_sub_sb_into_sb::<VBRANGE, REDUCE>, a, b, CHUNK);
    }

    #[inline(always)]
    fn va_neg_into_va<const CHUNK: usize, const VARANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &mut [u64]) {
        apply_v!(self, Self::sa_neg_into_sa::<VARANGE, REDUCE>, a, CHUNK);
    }

    #[inline(always)]
    fn va_neg_into_vb<const CHUNK: usize, const VARANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &mut [u64],
    ) {
        apply_vv!(self, Self::sa_neg_into_sb::<VARANGE, REDUCE>, a, b, CHUNK);
    }

    #[inline(always)]
    fn va_prep_mont_into_vb<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &mut [Montgomery<u64>],
    ) {
        apply_vv!(self, Self::sa_prep_mont_into_sb::<REDUCE>, a, b, CHUNK);
    }

    #[inline(always)]
    fn va_mont_mul_vb_into_vc<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[Montgomery<u64>],
        b: &[u64],
        c: &mut [u64],
    ) {
        apply_vvv!(self, Self::sa_mont_mul_sb_into_sc::<REDUCE>, a, b, c, CHUNK);
    }

    #[inline(always)]
    fn va_mont_mul_vb_into_vb<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[Montgomery<u64>],
        b: &mut [u64],
    ) {
        apply_vv!(self, Self::sa_mont_mul_sb_into_sb::<REDUCE>, a, b, CHUNK);
    }

    #[inline(always)]
    fn sa_barrett_mul_vb_into_vc<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &Barrett<u64>,
        b: &[u64],
        c: &mut [u64],
    ) {
        apply_svv!(
            self,
            Self::sa_barrett_mul_sb_into_sc::<REDUCE>,
            a,
            b,
            c,
            CHUNK
        );
    }

    #[inline(always)]
    fn sa_barrett_mul_vb_into_vb<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &Barrett<u64>,
        b: &mut [u64],
    ) {
        apply_sv!(self, Self::sa_barrett_mul_sb_into_sb::<REDUCE>, a, b, CHUNK);
    }

    fn va_sub_vb_mul_sc_into_vd<const CHUNK: usize, const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &[u64],
        c: &Barrett<u64>,
        d: &mut [u64],
    ) {
        apply_vvsv!(
            self,
            Self::sa_sub_sb_mul_sc_into_sd::<VBRANGE, REDUCE>,
            a,
            b,
            c,
            d,
            CHUNK
        );
    }

    fn va_sub_vb_mul_sc_into_vb<const CHUNK: usize, const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &Barrett<u64>,
        c: &mut [u64],
    ) {
        apply_vsv!(
            self,
            Self::sa_sub_sb_mul_sc_into_sb::<VBRANGE, REDUCE>,
            a,
            b,
            c,
            CHUNK
        );
    }

    // vec(a) <- (vec(a) + scalar(b)) * scalar(c);
    fn va_add_sb_mul_sc_into_va<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        b: &u64,
        c: &Barrett<u64>,
        a: &mut [u64],
    ) {
        apply_ssv!(
            self,
            Self::sa_add_sb_mul_sc_into_sa::<REDUCE>,
            b,
            c,
            a,
            CHUNK
        );
    }

    // vec(a) <- (vec(a) + scalar(b)) * scalar(c);
    fn va_add_sb_mul_sc_into_vd<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &u64,
        c: &Barrett<u64>,
        d: &mut [u64],
    ) {
        apply_vssv!(
            self,
            Self::sa_add_sb_mul_sc_into_sd::<REDUCE>,
            a,
            b,
            c,
            d,
            CHUNK
        );
    }
}
