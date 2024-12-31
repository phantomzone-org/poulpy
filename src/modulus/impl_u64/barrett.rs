use crate::modulus::barrett::{Barrett, BarrettPrecomp};
use crate::modulus::ReduceOnce;

use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;

impl BarrettPrecomp<u64>{
    
    pub fn new(q: u64) -> BarrettPrecomp<u64> {
        let big_r: BigUint = (BigUint::from(1 as usize)<<((u64::BITS<<1) as usize)) / BigUint::from(q);
        let lo: u64 = (&big_r & BigUint::from(u64::MAX)).to_u64().unwrap();
        let hi: u64 = (big_r >> u64::BITS).to_u64().unwrap();
        let mut precomp: BarrettPrecomp<u64> = Self{q, lo, hi, one:Barrett(0,0)};
        precomp.one = precomp.prepare(1);
        precomp
    }

    #[inline(always)]
    pub fn one(&self) -> Barrett<u64> {
        self.one
    }

    #[inline(always)]
    pub fn prepare(&self, v: u64) -> Barrett<u64> {
        debug_assert!(v < self.q);
        let quotient: u64 = (((v as u128) << 64) / self.q as u128) as _;
        Barrett(v, quotient)
    }

    /// Returns lhs mod q.
    #[inline(always)]
    pub fn reduce(&self, lhs: u64) -> u64{
        let mut r: u64 = self.reduce_lazy(lhs);
        r.reduce_once_assign(self.q);
        r
    }

    /// Returns lhs mod q in range [0, 2q-1].
    #[inline(always)]
    pub fn reduce_lazy(&self, lhs: u64) -> u64{
        let (_, mhi) = lhs.widening_mul(self.hi);
        lhs - mhi.wrapping_mul(self.q)
    }

    /// Assigns lhs mod q to lhs.
    #[inline(always)]
    pub fn reduce_assign(&self, lhs: &mut u64){
        self.reduce_lazy_assign(lhs);
        lhs.reduce_once_assign(self.q);
    }

    /// Assigns lhs mod q in range [0, 2q-1] to lhs.
    #[inline(always)]
    pub fn reduce_lazy_assign(&self, lhs: &mut u64){
        let (_, mhi) = lhs.widening_mul(self.hi);
        *lhs = *lhs - mhi.wrapping_mul(self.q)
    }

    #[inline(always)]
    pub fn mul_external(&self, lhs: Barrett<u64>, rhs: u64) -> u64 {
        let mut r: u64 = self.mul_external_lazy(lhs, rhs);
        r.reduce_once_assign(self.q);
        r
    }

    #[inline(always)]
    pub fn mul_external_assign(&self, lhs: Barrett<u64>, rhs: &mut u64){
        self.mul_external_lazy_assign(lhs, rhs);
        rhs.reduce_once_assign(self.q);
    }

    #[inline(always)]
    pub fn mul_external_lazy(&self, lhs: Barrett<u64>, rhs: u64) -> u64 {
        let mut r: u64 = rhs;
        self.mul_external_lazy_assign(lhs, &mut r);
        r
    }

    #[inline(always)]
    pub fn mul_external_lazy_assign(&self, lhs: Barrett<u64>, rhs: &mut u64){
        let t: u64 = ((*lhs.quotient() as u128 * *rhs as u128) >> 64) as _;
        *rhs = (rhs.wrapping_mul(*lhs.value())).wrapping_sub(self.q.wrapping_mul(t));
    }
}