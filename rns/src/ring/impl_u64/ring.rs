use crate::dft::ntt::Table;
use crate::modulus::barrett::Barrett;
use crate::modulus::montgomery::Montgomery;
use crate::modulus::prime::Prime;
use crate::modulus::{VectorOperations, ONCE};
use crate::modulus::{BARRETT, REDUCEMOD};
use crate::poly::Poly;
use crate::ring::Ring;
use crate::CHUNK;
use num_bigint::BigInt;
use num_traits::ToPrimitive;

impl Ring<u64> {
    pub fn new(n: usize, q_base: u64, q_power: usize) -> Self {
        let prime: Prime<u64> = Prime::<u64>::new(q_base, q_power);
        Self {
            n: n,
            modulus: prime.clone(),
            cyclotomic_order: n << 1,
            dft: Box::new(Table::<u64>::new(prime, n << 1)),
        }
    }

    pub fn from_bigint(&self, coeffs: &[BigInt], step: usize, a: &mut Poly<u64>) {
        assert!(
            step <= a.n(),
            "invalid step: step={} > a.n()={}",
            step,
            a.n()
        );
        assert!(
            coeffs.len() <= a.n() / step,
            "invalid coeffs: coeffs.len()={} > a.n()/step={}",
            coeffs.len(),
            a.n() / step
        );
        let q_big: BigInt = BigInt::from(self.modulus.q);
        a.0.iter_mut()
            .step_by(step)
            .enumerate()
            .for_each(|(i, v)| *v = (&coeffs[i] % &q_big).to_u64().unwrap());
    }
}

impl Ring<u64> {
    pub fn ntt_inplace<const LAZY: bool>(&self, poly: &mut Poly<u64>) {
        match LAZY {
            true => self.dft.forward_inplace_lazy(&mut poly.0),
            false => self.dft.forward_inplace(&mut poly.0),
        }
    }

    pub fn intt_inplace<const LAZY: bool>(&self, poly: &mut Poly<u64>) {
        match LAZY {
            true => self.dft.backward_inplace_lazy(&mut poly.0),
            false => self.dft.backward_inplace(&mut poly.0),
        }
    }

    pub fn ntt<const LAZY: bool>(&self, poly_in: &Poly<u64>, poly_out: &mut Poly<u64>) {
        poly_out.0.copy_from_slice(&poly_in.0);
        match LAZY {
            true => self.dft.forward_inplace_lazy(&mut poly_out.0),
            false => self.dft.forward_inplace(&mut poly_out.0),
        }
    }

    pub fn intt<const LAZY: bool>(&self, poly_in: &Poly<u64>, poly_out: &mut Poly<u64>) {
        poly_out.0.copy_from_slice(&poly_in.0);
        match LAZY {
            true => self.dft.backward_inplace_lazy(&mut poly_out.0),
            false => self.dft.backward_inplace(&mut poly_out.0),
        }
    }
}

impl Ring<u64> {
    #[inline(always)]
    pub fn a_add_b_into_b<const REDUCE: REDUCEMOD>(&self, a: &Poly<u64>, b: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus
            .va_add_vb_into_vb::<CHUNK, REDUCE>(&a.0, &mut b.0);
    }

    #[inline(always)]
    pub fn a_add_b_into_c<const REDUCE: REDUCEMOD>(
        &self,
        a: &Poly<u64>,
        b: &Poly<u64>,
        c: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        debug_assert!(c.n() == self.n(), "c.n()={} != n={}", c.n(), self.n());
        self.modulus
            .va_add_vb_into_vc::<CHUNK, REDUCE>(&a.0, &b.0, &mut c.0);
    }

    #[inline(always)]
    pub fn a_add_b_scalar_into_a<const REDUCE: REDUCEMOD>(&self, b: &u64, a: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        self.modulus.va_add_sb_into_va::<CHUNK, REDUCE>(b, &mut a.0);
    }

    #[inline(always)]
    pub fn a_add_b_scalar_into_c<const REDUCE: REDUCEMOD>(
        &self,
        a: &Poly<u64>,
        b: &u64,
        c: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(c.n() == self.n(), "c.n()={} != n={}", c.n(), self.n());
        self.modulus
            .va_add_sb_into_vc::<CHUNK, REDUCE>(&a.0, b, &mut c.0);
    }

    #[inline(always)]
    pub fn a_add_scalar_b_mul_c_scalar_barrett_into_a<const REDUCE: REDUCEMOD>(
        &self,
        b: &u64,
        c: &Barrett<u64>,
        a: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "b.n()={} != n={}", a.n(), self.n());
        self.modulus
            .va_add_sb_mul_sc_barrett_into_va::<CHUNK, REDUCE>(b, c, &mut a.0);
    }

    #[inline(always)]
    pub fn add_scalar_then_mul_scalar_barrett<const REDUCE: REDUCEMOD>(
        &self,
        a: &Poly<u64>,
        b: &u64,
        c: &Barrett<u64>,
        d: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(d.n() == self.n(), "c.n()={} != n={}", d.n(), self.n());
        self.modulus
            .va_add_sb_mul_sc_barrett_into_vd::<CHUNK, REDUCE>(&a.0, b, c, &mut d.0);
    }

    #[inline(always)]
    pub fn a_sub_b_into_b<const BRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &Poly<u64>,
        b: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus
            .va_sub_vb_into_vb::<CHUNK, BRANGE, REDUCE>(&a.0, &mut b.0);
    }

    #[inline(always)]
    pub fn a_sub_b_into_a<const BRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        b: &Poly<u64>,
        a: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus
            .va_sub_vb_into_va::<CHUNK, BRANGE, REDUCE>(&b.0, &mut a.0);
    }

    #[inline(always)]
    pub fn a_sub_b_into_c<const BRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &Poly<u64>,
        b: &Poly<u64>,
        c: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        debug_assert!(c.n() == self.n(), "c.n()={} != n={}", c.n(), self.n());
        self.modulus
            .va_sub_vb_into_vc::<CHUNK, BRANGE, REDUCE>(&a.0, &b.0, &mut c.0);
    }

    #[inline(always)]
    pub fn a_neg_into_b<const ARANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &Poly<u64>,
        b: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus
            .va_neg_into_vb::<CHUNK, ARANGE, REDUCE>(&a.0, &mut b.0);
    }

    #[inline(always)]
    pub fn a_neg_into_a<const ARANGE: u8, const REDUCE: REDUCEMOD>(&self, a: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        self.modulus
            .va_neg_into_va::<CHUNK, ARANGE, REDUCE>(&mut a.0);
    }

    #[inline(always)]
    pub fn a_prepare_montgomery_into_a<const REDUCE: REDUCEMOD>(
        &self,
        a: &mut Poly<Montgomery<u64>>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        self.modulus
            .va_prepare_montgomery_into_va::<CHUNK, REDUCE>(&mut a.0);
    }

    #[inline(always)]
    pub fn a_mul_b_montgomery_into_c<const REDUCE: REDUCEMOD>(
        &self,
        a: &Poly<Montgomery<u64>>,
        b: &Poly<u64>,
        c: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        debug_assert!(c.n() == self.n(), "c.n()={} != n={}", c.n(), self.n());
        self.modulus
            .va_mul_vb_montgomery_into_vc::<CHUNK, REDUCE>(&a.0, &b.0, &mut c.0);
    }

    #[inline(always)]
    pub fn a_mul_b_montgomery_add_c_into_c<const REDUCE1: REDUCEMOD, const REDUCE2: REDUCEMOD>(
        &self,
        a: &Poly<Montgomery<u64>>,
        b: &Poly<u64>,
        c: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        debug_assert!(c.n() == self.n(), "c.n()={} != n={}", c.n(), self.n());
        self.modulus
            .va_mul_vb_montgomery_add_vc_into_vc::<CHUNK, REDUCE1, REDUCE2>(&a.0, &b.0, &mut c.0);
    }

    #[inline(always)]
    pub fn a_mul_b_montgomery_into_a<const REDUCE: REDUCEMOD>(
        &self,
        b: &Poly<Montgomery<u64>>,
        a: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus
            .va_mul_vb_montgomery_into_va::<CHUNK, REDUCE>(&b.0, &mut a.0);
    }

    #[inline(always)]
    pub fn a_mul_b_scalar_into_c<const REDUCE: REDUCEMOD>(
        &self,
        a: &Poly<u64>,
        b: &u64,
        c: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(c.n() == self.n(), "c.n()={} != n={}", c.n(), self.n());
        self.modulus.va_mul_sb_barrett_into_vc::<CHUNK, REDUCE>(
            &a.0,
            &self.modulus.barrett.prepare(*b),
            &mut c.0,
        );
    }

    #[inline(always)]
    pub fn a_mul_b_scalar_into_a<const REDUCE: REDUCEMOD>(&self, b: &u64, a: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        self.modulus.va_mul_sb_barrett_into_va::<CHUNK, REDUCE>(
            &self
                .modulus
                .barrett
                .prepare(self.modulus.barrett.reduce::<BARRETT>(b)),
            &mut a.0,
        );
    }

    #[inline(always)]
    pub fn a_mul_b_scalar_barrett_into_a<const REDUCE: REDUCEMOD>(
        &self,
        b: &Barrett<u64>,
        a: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        self.modulus
            .va_mul_sb_barrett_into_va::<CHUNK, REDUCE>(b, &mut a.0);
    }

    #[inline(always)]
    pub fn a_mul_b_scalar_barrett_into_c<const REDUCE: REDUCEMOD>(
        &self,
        a: &Poly<u64>,
        b: &Barrett<u64>,
        c: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        self.modulus
            .va_mul_sb_barrett_into_vc::<CHUNK, REDUCE>(&a.0, b, &mut c.0);
    }

    #[inline(always)]
    pub fn a_sub_b_mul_c_scalar_barrett_into_d<const VBRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &Poly<u64>,
        b: &Poly<u64>,
        c: &Barrett<u64>,
        d: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        debug_assert!(d.n() == self.n(), "d.n()={} != n={}", d.n(), self.n());
        self.modulus
            .va_sub_vb_mul_sc_barrett_into_vd::<CHUNK, VBRANGE, REDUCE>(&a.0, &b.0, c, &mut d.0);
    }

    #[inline(always)]
    pub fn b_sub_a_mul_c_scalar_barrett_into_a<const BRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        b: &Poly<u64>,
        c: &Barrett<u64>,
        a: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus
            .va_sub_vb_mul_sc_barrett_into_vb::<CHUNK, BRANGE, REDUCE>(&b.0, c, &mut a.0);
    }

    #[inline(always)]
    pub fn a_sub_b_add_c_scalar_mul_d_scalar_barrett_into_e<
        const BRANGE: u8,
        const REDUCE: REDUCEMOD,
    >(
        &self,
        a: &Poly<u64>,
        b: &Poly<u64>,
        c: &u64,
        d: &Barrett<u64>,
        e: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        debug_assert!(e.n() == self.n(), "e.n()={} != n={}", e.n(), self.n());
        self.modulus
            .vb_sub_va_add_sc_mul_sd_barrett_into_ve::<CHUNK, BRANGE, REDUCE>(
                &a.0, &b.0, c, d, &mut e.0,
            );
    }

    #[inline(always)]
    pub fn b_sub_a_add_c_scalar_mul_d_scalar_barrett_into_a<
        const BRANGE: u8,
        const REDUCE: REDUCEMOD,
    >(
        &self,
        b: &Poly<u64>,
        c: &u64,
        d: &Barrett<u64>,
        a: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus
            .vb_sub_va_add_sc_mul_sd_barrett_into_va::<CHUNK, BRANGE, REDUCE>(&b.0, c, d, &mut a.0);
    }

    pub fn a_rsh_scalar_b_mask_scalar_c_into_d(
        &self,
        a: &Poly<u64>,
        b: &usize,
        c: &u64,
        d: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(d.n() == self.n(), "d.n()={} != n={}", d.n(), self.n());
        self.modulus
            .va_rsh_sb_mask_sc_into_vd::<CHUNK>(&a.0, b, c, &mut d.0);
    }

    pub fn a_rsh_scalar_b_mask_scalar_c_add_d_into_d(
        &self,
        a: &Poly<u64>,
        b: &usize,
        c: &u64,
        d: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(d.n() == self.n(), "d.n()={} != n={}", d.n(), self.n());
        self.modulus
            .va_rsh_sb_mask_sc_add_vd_into_vd::<CHUNK>(&a.0, b, c, &mut d.0);
    }

    pub fn a_ith_digit_unsigned_base_scalar_b_into_c(
        &self,
        i: usize,
        a: &Poly<u64>,
        b: &usize,
        c: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(c.n() == self.n(), "c.n()={} != n={}", c.n(), self.n());
        self.modulus
            .va_ith_digit_unsigned_base_sb_into_vc::<CHUNK>(i, &a.0, b, &mut c.0);
    }

    pub fn a_ith_digit_signed_base_scalar_b_into_c<const BALANCED: bool>(
        &self,
        i: usize,
        a: &Poly<u64>,
        b: &usize,
        carry: &mut Poly<u64>,
        c: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(
            carry.n() == self.n(),
            "carry.n()={} != n={}",
            carry.n(),
            self.n()
        );
        debug_assert!(c.n() == self.n(), "c.n()={} != n={}", c.n(), self.n());
        self.modulus
            .va_ith_digit_signed_base_sb_into_vc::<CHUNK, BALANCED>(
                i,
                &a.0,
                b,
                &mut carry.0,
                &mut c.0,
            );
    }

    pub fn a_mul_by_x_pow_b_into_a(&self, b: i32, a: &mut Poly<u64>) {
        let n: usize = self.n();
        let cyclotomic_order: usize = self.cyclotomic_order();

        let b_0: usize = (b as usize).wrapping_add(cyclotomic_order) & (cyclotomic_order - 1);
        let b_1: usize = b as usize & (n - 1);

        a.0.rotate_right(b_1);

        if b_0 > b_1 {
            self.modulus
                .va_neg_into_va::<CHUNK, 1, ONCE>(&mut a.0[b_1..])
        } else {
            self.modulus
                .va_neg_into_va::<CHUNK, 1, ONCE>(&mut a.0[..b_1])
        }
    }
}
