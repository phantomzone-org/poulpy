use crate::modulus::barrett::Barrett;
use crate::modulus::montgomery::Montgomery;
use crate::modulus::prime::Prime;
use crate::modulus::{ScalarOperations, VectorOperations};
use crate::modulus::{NONE, REDUCEMOD};
use crate::{
    apply_ssv, apply_sv, apply_v, apply_vsssvv, apply_vssv, apply_vsv, apply_vv, apply_vvssv,
    apply_vvsv, apply_vvv,
};
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
    fn sa_sub_sb_into_sc<const SBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        b: &u64,
        c: &mut u64,
    ) {
        match SBRANGE {
            1 => *c = *a + self.q - *b,
            2 => *c = *a + self.two_q - *b,
            4 => *c = *a + self.four_q - *b,
            _ => unreachable!("invalid SBRANGE argument"),
        }
        self.sa_reduce_into_sa::<REDUCE>(c)
    }

    #[inline(always)]
    fn sa_sub_sb_into_sa<const SBRANGE: u8, const REDUCE: REDUCEMOD>(&self, b: &u64, a: &mut u64) {
        match SBRANGE {
            1 => *a = *a + self.q - *b,
            2 => *a = *a + self.two_q - *b,
            4 => *a = *a + self.four_q - *b,
            _ => unreachable!("invalid SBRANGE argument"),
        }
        self.sa_reduce_into_sa::<REDUCE>(a)
    }

    #[inline(always)]
    fn sa_sub_sb_into_sb<const SBRANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &u64, b: &mut u64) {
        match SBRANGE {
            1 => *b = *a + self.q - *b,
            2 => *b = *a + self.two_q - *b,
            4 => *b = *a + self.four_q - *b,
            _ => unreachable!("invalid SBRANGE argument"),
        }
        self.sa_reduce_into_sa::<REDUCE>(b)
    }

    #[inline(always)]
    fn sa_neg_into_sa<const SBRANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &mut u64) {
        match SBRANGE {
            1 => *a = self.q - *a,
            2 => *a = self.two_q - *a,
            4 => *a = self.four_q - *a,
            _ => unreachable!("invalid SBRANGE argument"),
        }
        self.sa_reduce_into_sa::<REDUCE>(a)
    }

    #[inline(always)]
    fn sa_neg_into_sb<const SBRANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &u64, b: &mut u64) {
        match SBRANGE {
            1 => *b = self.q - *a,
            2 => *b = self.two_q - *a,
            4 => *b = self.four_q - *a,
            _ => unreachable!("invalid SBRANGE argument"),
        }
        self.sa_reduce_into_sa::<REDUCE>(b)
    }

    #[inline(always)]
    fn sa_prepare_montgomery_into_sa<const REDUCE: REDUCEMOD>(&self, a: &mut Montgomery<u64>) {
        *a = self.montgomery.prepare::<REDUCE>(*a);
    }

    #[inline(always)]
    fn sa_prepare_montgomery_into_sb<const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        b: &mut Montgomery<u64>,
    ) {
        self.montgomery.prepare_assign::<REDUCE>(*a, b);
    }

    #[inline(always)]
    fn sa_mul_sb_montgomery_into_sc<const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        b: &Montgomery<u64>,
        c: &mut u64,
    ) {
        *c = self.montgomery.mul_external::<REDUCE>(*a, *b);
    }

    #[inline(always)]
    fn sa_mul_sb_montgomery_add_sc_into_sc<const REDUCE1: REDUCEMOD, const REDUCE2: REDUCEMOD>(
        &self,
        a: &u64,
        b: &Montgomery<u64>,
        c: &mut u64,
    ) {
        *c += self.montgomery.mul_external::<REDUCE1>(*a, *b);
        self.sa_reduce_into_sa::<REDUCE2>(c);
    }

    #[inline(always)]
    fn sa_mul_sb_montgomery_into_sa<const REDUCE: REDUCEMOD>(
        &self,
        b: &Montgomery<u64>,
        a: &mut u64,
    ) {
        self.montgomery.mul_external_assign::<REDUCE>(*b, a);
    }

    #[inline(always)]
    fn sa_mul_sb_barrett_into_sc<const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        b: &Barrett<u64>,
        c: &mut u64,
    ) {
        *c = self.barrett.mul_external::<REDUCE>(b, a);
    }

    #[inline(always)]
    fn sa_mul_sb_barrett_into_sa<const REDUCE: REDUCEMOD>(&self, b: &Barrett<u64>, a: &mut u64) {
        self.barrett.mul_external_assign::<REDUCE>(b, a);
    }

    #[inline(always)]
    fn sa_sub_sb_mul_sc_barrett_into_sd<const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        b: &u64,
        c: &Barrett<u64>,
        d: &mut u64,
    ) {
        match VBRANGE {
            1 => *d = a + self.q - b,
            2 => *d = a + self.two_q - b,
            4 => *d = a + self.four_q - b,
            _ => unreachable!("invalid SBRANGE argument"),
        }
        self.barrett.mul_external_assign::<REDUCE>(c, d);
    }

    #[inline(always)]
    fn sa_sub_sb_mul_sc_barrett_into_sb<const SBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        c: &Barrett<u64>,
        b: &mut u64,
    ) {
        self.sa_sub_sb_into_sb::<SBRANGE, NONE>(a, b);
        self.barrett.mul_external_assign::<REDUCE>(c, b);
    }

    #[inline(always)]
    fn sa_add_sb_mul_sc_barrett_into_sd<const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        b: &u64,
        c: &Barrett<u64>,
        d: &mut u64,
    ) {
        *d = self.barrett.mul_external::<REDUCE>(c, &(*a + b));
    }

    #[inline(always)]
    fn sa_add_sb_mul_sc_barrett_into_sa<const REDUCE: REDUCEMOD>(
        &self,
        b: &u64,
        c: &Barrett<u64>,
        a: &mut u64,
    ) {
        *a = self.barrett.mul_external::<REDUCE>(c, &(*a + b));
    }

    #[inline(always)]
    fn sb_sub_sa_add_sc_mul_sd_barrett_into_se<const SBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &u64,
        b: &u64,
        c: &u64,
        d: &Barrett<u64>,
        e: &mut u64,
    ) {
        self.sa_sub_sb_into_sc::<SBRANGE, NONE>(&(b + c), a, e);
        self.barrett.mul_external_assign::<REDUCE>(d, e);
    }

    #[inline(always)]
    fn sb_sub_sa_add_sc_mul_sd_barrett_into_sa<const SBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        b: &u64,
        c: &u64,
        d: &Barrett<u64>,
        a: &mut u64,
    ) {
        self.sa_sub_sb_into_sb::<SBRANGE, NONE>(&(b + c), a);
        self.barrett.mul_external_assign::<REDUCE>(d, a);
    }

    #[inline(always)]
    fn sa_rsh_sb_mask_sc_into_sa(&self, b: &usize, c: &u64, a: &mut u64) {
        *a = (*a >> b) & c
    }

    #[inline(always)]
    fn sa_rsh_sb_mask_sc_into_sd(&self, a: &u64, b: &usize, c: &u64, d: &mut u64) {
        *d = (*a >> b) & c
    }

    #[inline(always)]
    fn sa_rsh_sb_mask_sc_add_sd_into_sd(&self, a: &u64, b: &usize, c: &u64, d: &mut u64) {
        *d += (*a >> b) & c
    }

    #[inline(always)]
    fn sa_signed_digit_into_sb<const CARRYOVERWRITE: bool, const BALANCED: bool>(
        &self,
        a: &u64,
        base: &u64,
        shift: &usize,
        mask: &u64,
        carry: &mut u64,
        b: &mut u64,
    ) {
        if CARRYOVERWRITE {
            self.sa_rsh_sb_mask_sc_into_sd(a, shift, mask, carry);
        } else {
            self.sa_rsh_sb_mask_sc_add_sd_into_sd(a, shift, mask, carry);
        }

        let c: u64 = if BALANCED && *carry == base >> 1 {
            a & 1
        } else {
            ((*carry | (*carry << 1)) >> shift) & 1
        };

        *b = *carry + (self.q - base) * c;
        *carry = c;
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
        apply_vvv!(
            self,
            Self::sa_sub_sb_into_sc::<VBRANGE, REDUCE>,
            a,
            b,
            c,
            CHUNK
        );
    }

    #[inline(always)]
    fn va_sub_vb_into_va<const CHUNK: usize, const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        b: &[u64],
        a: &mut [u64],
    ) {
        apply_vv!(
            self,
            Self::sa_sub_sb_into_sa::<VBRANGE, REDUCE>,
            b,
            a,
            CHUNK
        );
    }

    #[inline(always)]
    fn va_sub_vb_into_vb<const CHUNK: usize, const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &mut [u64],
    ) {
        apply_vv!(
            self,
            Self::sa_sub_sb_into_sb::<VBRANGE, REDUCE>,
            a,
            b,
            CHUNK
        );
    }

    #[inline(always)]
    fn va_neg_into_va<const CHUNK: usize, const VARANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &mut [u64],
    ) {
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
        apply_vv!(
            self,
            Self::sa_prepare_montgomery_into_sb::<REDUCE>,
            a,
            b,
            CHUNK
        );
    }

    #[inline(always)]
    fn va_prepare_montgomery_into_va<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &mut [Montgomery<u64>],
    ) {
        apply_v!(
            self,
            Self::sa_prepare_montgomery_into_sa::<REDUCE>,
            a,
            CHUNK
        );
    }

    #[inline(always)]
    fn va_mul_vb_montgomery_into_vc<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[Montgomery<u64>],
        b: &[u64],
        c: &mut [u64],
    ) {
        apply_vvv!(
            self,
            Self::sa_mul_sb_montgomery_into_sc::<REDUCE>,
            a,
            b,
            c,
            CHUNK
        );
    }

    #[inline(always)]
    fn va_mul_vb_montgomery_add_vc_into_vc<
        const CHUNK: usize,
        const REDUCE1: REDUCEMOD,
        const REDUCE2: REDUCEMOD,
    >(
        &self,
        a: &[Montgomery<u64>],
        b: &[u64],
        c: &mut [u64],
    ) {
        apply_vvv!(
            self,
            Self::sa_mul_sb_montgomery_add_sc_into_sc::<REDUCE1, REDUCE2>,
            a,
            b,
            c,
            CHUNK
        );
    }

    #[inline(always)]
    fn va_mul_vb_montgomery_into_va<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        b: &[Montgomery<u64>],
        a: &mut [u64],
    ) {
        apply_vv!(
            self,
            Self::sa_mul_sb_montgomery_into_sa::<REDUCE>,
            b,
            a,
            CHUNK
        );
    }

    #[inline(always)]
    fn va_mul_sb_barrett_into_vc<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &Barrett<u64>,
        c: &mut [u64],
    ) {
        apply_vsv!(
            self,
            Self::sa_mul_sb_barrett_into_sc::<REDUCE>,
            a,
            b,
            c,
            CHUNK
        );
    }

    #[inline(always)]
    fn va_mul_sb_barrett_into_va<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        b: &Barrett<u64>,
        a: &mut [u64],
    ) {
        apply_sv!(self, Self::sa_mul_sb_barrett_into_sa::<REDUCE>, b, a, CHUNK);
    }

    fn va_sub_vb_mul_sc_barrett_into_vd<
        const CHUNK: usize,
        const VBRANGE: u8,
        const REDUCE: REDUCEMOD,
    >(
        &self,
        a: &[u64],
        b: &[u64],
        c: &Barrett<u64>,
        d: &mut [u64],
    ) {
        apply_vvsv!(
            self,
            Self::sa_sub_sb_mul_sc_barrett_into_sd::<VBRANGE, REDUCE>,
            a,
            b,
            c,
            d,
            CHUNK
        );
    }

    fn va_sub_vb_mul_sc_barrett_into_vb<
        const CHUNK: usize,
        const VBRANGE: u8,
        const REDUCE: REDUCEMOD,
    >(
        &self,
        a: &[u64],
        b: &Barrett<u64>,
        c: &mut [u64],
    ) {
        apply_vsv!(
            self,
            Self::sa_sub_sb_mul_sc_barrett_into_sb::<VBRANGE, REDUCE>,
            a,
            b,
            c,
            CHUNK
        );
    }

    // vec(a) <- (vec(a) + scalar(b)) * scalar(c);
    fn va_add_sb_mul_sc_barrett_into_va<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        b: &u64,
        c: &Barrett<u64>,
        a: &mut [u64],
    ) {
        apply_ssv!(
            self,
            Self::sa_add_sb_mul_sc_barrett_into_sa::<REDUCE>,
            b,
            c,
            a,
            CHUNK
        );
    }

    // vec(a) <- (vec(a) + scalar(b)) * scalar(c);
    fn va_add_sb_mul_sc_barrett_into_vd<const CHUNK: usize, const REDUCE: REDUCEMOD>(
        &self,
        a: &[u64],
        b: &u64,
        c: &Barrett<u64>,
        d: &mut [u64],
    ) {
        apply_vssv!(
            self,
            Self::sa_add_sb_mul_sc_barrett_into_sd::<REDUCE>,
            a,
            b,
            c,
            d,
            CHUNK
        );
    }

    // vec(e) <- (vec(a) - vec(b) + scalar(c)) * scalar(e).
    fn vb_sub_va_add_sc_mul_sd_barrett_into_ve<
        const CHUNK: usize,
        const VBRANGE: u8,
        const REDUCE: REDUCEMOD,
    >(
        &self,
        va: &[u64],
        vb: &[u64],
        sc: &u64,
        sd: &Barrett<u64>,
        ve: &mut [u64],
    ) {
        apply_vvssv!(
            self,
            Self::sb_sub_sa_add_sc_mul_sd_barrett_into_se::<VBRANGE, REDUCE>,
            va,
            vb,
            sc,
            sd,
            ve,
            CHUNK
        );
    }

    // vec(a) <- (vec(b) - vec(a) + scalar(c)) * scalar(e).
    fn vb_sub_va_add_sc_mul_sd_barrett_into_va<
        const CHUNK: usize,
        const VBRANGE: u8,
        const REDUCE: REDUCEMOD,
    >(
        &self,
        vb: &[u64],
        sc: &u64,
        sd: &Barrett<u64>,
        va: &mut [u64],
    ) {
        apply_vssv!(
            self,
            Self::sb_sub_sa_add_sc_mul_sd_barrett_into_sa::<VBRANGE, REDUCE>,
            vb,
            sc,
            sd,
            va,
            CHUNK
        );
    }

    // vec(a) <- (vec(a)>>scalar(b)) & scalar(c).
    fn va_rsh_sb_mask_sc_into_va<const CHUNK: usize>(&self, sb: &usize, sc: &u64, va: &mut [u64]) {
        apply_ssv!(self, Self::sa_rsh_sb_mask_sc_into_sa, sb, sc, va, CHUNK);
    }

    // vec(d) <- (vec(a)>>scalar(b)) & scalar(c).
    fn va_rsh_sb_mask_sc_into_vd<const CHUNK: usize>(
        &self,
        va: &[u64],
        sb: &usize,
        sc: &u64,
        vd: &mut [u64],
    ) {
        apply_vssv!(self, Self::sa_rsh_sb_mask_sc_into_sd, va, sb, sc, vd, CHUNK);
    }

    // vec(d) <- vec(d) + (vec(a)>>scalar(b)) & scalar(c).
    fn va_rsh_sb_mask_sc_add_vd_into_vd<const CHUNK: usize>(
        &self,
        va: &[u64],
        sb: &usize,
        sc: &u64,
        vd: &mut [u64],
    ) {
        apply_vssv!(
            self,
            Self::sa_rsh_sb_mask_sc_add_sd_into_sd,
            va,
            sb,
            sc,
            vd,
            CHUNK
        );
    }

    // vec(c) <- i-th unsigned digit base 2^{sb} of vec(a).
    // vec(c) is ensured to be in the range [0, 2^{sb}-1[ with E[vec(c)] = 2^{sb}-1.
    fn va_ith_digit_unsigned_base_sb_into_vc<const CHUNK: usize>(
        &self,
        i: usize,
        va: &[u64],
        sb: &usize,
        vc: &mut [u64],
    ) {
        self.va_rsh_sb_mask_sc_into_vd::<CHUNK>(va, &(i * sb), &((1 << sb) - 1), vc);
    }

    // vec(c) <- i-th signed digit base 2^{w} of vec(a).
    // Reads the carry of the i-1-th iteration and write the carry on the i-th iteration on carry.
    // if i > 0, carry of the i-1th iteration must be provided.
    // if BALANCED: vec(c) is ensured to be [-2^{sb-1}, 2^{sb-1}[ with E[vec(c)] = 0, else E[vec(c)] = -0.5
    fn va_ith_digit_signed_base_sb_into_vc<const CHUNK: usize, const BALANCED: bool>(
        &self,
        i: usize,
        va: &[u64],
        sb: &usize,
        carry: &mut [u64],
        vc: &mut [u64],
    ) {
        let base: u64 = 1 << sb;
        let mask: u64 = base - 1;
        if i == 0 {
            apply_vsssvv!(
                self,
                Self::sa_signed_digit_into_sb::<true, BALANCED>,
                va,
                &base,
                &(i * sb),
                &mask,
                carry,
                vc,
                CHUNK
            );
        } else {
            apply_vsssvv!(
                self,
                Self::sa_signed_digit_into_sb::<false, BALANCED>,
                va,
                &base,
                &(i * sb),
                &mask,
                carry,
                vc,
                CHUNK
            );
        }
    }
}
