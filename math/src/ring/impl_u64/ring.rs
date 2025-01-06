use crate::dft::ntt::Table;
use crate::modulus::barrett::Barrett;
use crate::modulus::montgomery::Montgomery;
use crate::modulus::prime::Prime;
use crate::modulus::VectorOperations;
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
            dft: Box::new(Table::<u64>::new(prime, (2 * n) as u64)),
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
    pub fn add_inplace<const REDUCE: REDUCEMOD>(&self, a: &Poly<u64>, b: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus
            .va_add_vb_into_vb::<CHUNK, REDUCE>(&a.0, &mut b.0);
    }

    #[inline(always)]
    pub fn add<const REDUCE: REDUCEMOD>(&self, a: &Poly<u64>, b: &Poly<u64>, c: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        debug_assert!(c.n() == self.n(), "c.n()={} != n={}", c.n(), self.n());
        self.modulus
            .va_add_vb_into_vc::<CHUNK, REDUCE>(&a.0, &b.0, &mut c.0);
    }

    #[inline(always)]
    pub fn add_scalar_inplace<const REDUCE: REDUCEMOD>(&self, b: &u64, a: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        self.modulus.va_add_sb_into_va::<CHUNK, REDUCE>(b, &mut a.0);
    }

    #[inline(always)]
    pub fn add_scalar<const REDUCE: REDUCEMOD>(&self, a: &Poly<u64>, b: &u64, c: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(c.n() == self.n(), "c.n()={} != n={}", c.n(), self.n());
        self.modulus
            .va_add_sb_into_vc::<CHUNK, REDUCE>(&a.0, b, &mut c.0);
    }

    #[inline(always)]
    pub fn add_scalar_then_mul_scalar_barrett_inplace<const REDUCE: REDUCEMOD>(&self, b: &u64, c: &Barrett<u64>, a: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "b.n()={} != n={}", a.n(), self.n());
        self.modulus.va_add_sb_mul_sc_into_va::<CHUNK, REDUCE>(b, c, &mut a.0);
    }

    #[inline(always)]
    pub fn add_scalar_then_mul_scalar_barrett<const REDUCE: REDUCEMOD>(&self, a: &Poly<u64>, b: &u64, c: &Barrett<u64>, d: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(d.n() == self.n(), "c.n()={} != n={}", d.n(), self.n());
        self.modulus
            .va_add_sb_mul_sc_into_vd::<CHUNK, REDUCE>(&a.0, b, c, &mut d.0);
    }

    #[inline(always)]
    pub fn sub_inplace<const BRANGE:u8, const REDUCE: REDUCEMOD>(&self, a: &Poly<u64>, b: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus
            .va_sub_vb_into_vb::<CHUNK, BRANGE, REDUCE>(&a.0, &mut b.0);
    }

    #[inline(always)]
    pub fn sub<const BRANGE:u8, const REDUCE: REDUCEMOD>(&self, a: &Poly<u64>, b: &Poly<u64>, c: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        debug_assert!(c.n() == self.n(), "c.n()={} != n={}", c.n(), self.n());
        self.modulus
            .va_sub_vb_into_vc::<CHUNK, BRANGE, REDUCE>(&a.0, &b.0, &mut c.0);
    }

    #[inline(always)]
    pub fn neg<const ARANGE:u8, const REDUCE: REDUCEMOD>(&self, a: &Poly<u64>, b: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus.va_neg_into_vb::<CHUNK, ARANGE, REDUCE>(&a.0, &mut b.0);
    }

    #[inline(always)]
    pub fn neg_inplace<const ARANGE:u8,const REDUCE: REDUCEMOD>(&self, a: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        self.modulus.va_neg_into_va::<CHUNK, ARANGE, REDUCE>(&mut a.0);
    }

    #[inline(always)]
    pub fn mul_montgomery_external<const REDUCE: REDUCEMOD>(
        &self,
        a: &Poly<Montgomery<u64>>,
        b: &Poly<u64>,
        c: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        debug_assert!(c.n() == self.n(), "c.n()={} != n={}", c.n(), self.n());
        self.modulus
            .va_mont_mul_vb_into_vc::<CHUNK, REDUCE>(&a.0, &b.0, &mut c.0);
    }

    #[inline(always)]
    pub fn mul_montgomery_external_inplace<const REDUCE: REDUCEMOD>(
        &self,
        a: &Poly<Montgomery<u64>>,
        b: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus
            .va_mont_mul_vb_into_vb::<CHUNK, REDUCE>(&a.0, &mut b.0);
    }

    #[inline(always)]
    pub fn mul_scalar<const REDUCE: REDUCEMOD>(&self, a: &Poly<u64>, b: &u64, c: &mut Poly<u64>) {
        debug_assert!(a.n() == self.n(), "b.n()={} != n={}", a.n(), self.n());
        debug_assert!(c.n() == self.n(), "c.n()={} != n={}", c.n(), self.n());
        self.modulus.sa_barrett_mul_vb_into_vc::<CHUNK, REDUCE>(
            &self.modulus.barrett.prepare(*b),
            &a.0,
            &mut c.0,
        );
    }

    #[inline(always)]
    pub fn mul_scalar_inplace<const REDUCE: REDUCEMOD>(&self, a: &u64, b: &mut Poly<u64>) {
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus.sa_barrett_mul_vb_into_vb::<CHUNK, REDUCE>(
            &self
                .modulus
                .barrett
                .prepare(self.modulus.barrett.reduce::<BARRETT>(a)),
            &mut b.0,
        );
    }

    #[inline(always)]
    pub fn mul_scalar_barrett_inplace<const REDUCE: REDUCEMOD>(
        &self,
        a: &Barrett<u64>,
        b: &mut Poly<u64>,
    ) {
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus
            .sa_barrett_mul_vb_into_vb::<CHUNK, REDUCE>(a, &mut b.0);
    }

    #[inline(always)]
    pub fn mul_scalar_barrett<const REDUCE: REDUCEMOD>(
        &self,
        a: &Barrett<u64>,
        b: &Poly<u64>,
        c: &mut Poly<u64>,
    ) {
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus
            .sa_barrett_mul_vb_into_vc::<CHUNK, REDUCE>(a, &b.0, &mut c.0);
    }

    #[inline(always)]
    pub fn a_sub_b_mul_c_scalar_barrett<const VBRANGE: u8, const REDUCE: REDUCEMOD>(
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
            .va_sub_vb_mul_sc_into_vd::<CHUNK, VBRANGE, REDUCE>(&a.0, &b.0, c, &mut d.0);
    }

    #[inline(always)]
    pub fn a_sub_b_mul_c_scalar_barrett_inplace<const BRANGE: u8, const REDUCE: REDUCEMOD>(
        &self,
        a: &Poly<u64>,
        c: &Barrett<u64>,
        b: &mut Poly<u64>,
    ) {
        debug_assert!(a.n() == self.n(), "a.n()={} != n={}", a.n(), self.n());
        debug_assert!(b.n() == self.n(), "b.n()={} != n={}", b.n(), self.n());
        self.modulus
            .va_sub_vb_mul_sc_into_vb::<CHUNK, BRANGE, REDUCE>(&a.0, c, &mut b.0);
    }
}
